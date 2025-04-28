import asyncio
import os
from typing import Any, Optional

import polars as pl
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

API_KEY = os.getenv('OPENAI_API_KEY')
SYSTEM_PROMPT = """
# Task
The user will provide InterProScan results for a gene. 
Summarize the gene's main function in English, using no more than 20 words based on InterPro or similar annotation. 
If no reliable information is available, write "unknown". Follow established nomenclature practices; provide the gene or protein name first if it can be determined, followed by the concise function.

具体逻辑如下：
| 优先级 | 注释内容                                                         | 理由               |
| --- | ------------------------------------------------------------ | ---------------- |
| 1   | **具体的功能蛋白名**（如"ATP synthase subunit beta"）                   | 直接可理解，方便后续基因功能分类 |
| 2   | **通路成员**（如"Photosystem II oxygen-evolving enhancer protein"） | 可推断生物过程          |
| 3   | **家族成员**（如"ABC transporter family member"）                   | 有参考意义，但较宽泛       |
| 4   | **domain-only**（如"TPR domain"）                               | 不够具体，仅限于结构推测     |
| 5   | **未知/预测/重复/无意义注释**                                           | 排除               |
总结成一句话就是：**优先选蛋白功能而不是结构描述，能标具体的就不留宽泛的。**

# Input Format
用户给出的注释信息包括三个部分，结构如下：
```
"Function description" [Preferred Name] (Source)
```
其中Preferred Name可能为空。

## Example
input:
```
Myb DNA-binding like [None] (eggNOG-mapper)
post-chaperonin tubulin folding pathway [TBCA] (eggNOG-mapper)
Tubulin binding cofactor A superfamily [None] (InterProScan)
Tubulin binding cofactor A [None] (InterProScan)
```

output:
- name: TBCA
- description: Assists post-chaperonin tubulin folding
"""

HUMAN_PROMPT = """下面每一行代表该基因的一个注释结果。
{input}
"""

llm = ChatOpenAI(model='o4-mini', base_url='https://aihubmix.com/v1', api_key=API_KEY)


# 结构定义在这里，直接修改不同词条后面的description就可以
class Annotate(BaseModel):
    name: str = Field(description='Gene/protein name')
    description: str = Field(description="gene's main function in English ≤20 words")


def load_data(inter_pro_file: str, eggnog_file: str, output_file: str) -> pl.DataFrame:
    """加载数据并过滤

    Args:
        inter_pro_file: InterProScan原始输入（TSV）
        eggnog_file: eggNOG原始输入（TSV）
        output_file: 用于暂存中间结果的文件（CSV）

    Returns:
        过滤后的dataframe
    """

    # InterProScan结果读取
    interpro_df = pl.read_csv(
        inter_pro_file,
        has_header=False,
        quote_char=None,
        separator='\t',
        null_values='-',
        new_columns=[
            'ID',
            'Sequence MD5',
            'Sequence length',
            'Analysis',
            'Signature accession',
            'Signature description',
            'Start location',
            'Stop location',
            'Score',
            'Status',
            'Date',
            'InterPro annotations accession',
            'InterPro annotations description',
            'GO annotations',
            'Pathways annotations',
        ],
    )
    logger.info(f'InterProScan数据读取完毕，共{interpro_df.shape[0]}条')

    filter_interpro_df = (
        interpro_df.select(
            [
                'ID',
                'Score',
                'InterPro annotations accession',
                'InterPro annotations description',
            ]
        )
        .with_columns(pl.col('ID').str.slice(0, pl.col('ID').str.len_chars() - 2))
        .rename(
            {
                'InterPro annotations accession': 'Accession',
                'InterPro annotations description': 'Description',
            }
        )
        .filter(pl.col('Description').is_not_null())
        .unique()
        # .with_columns(pl.col('Score').fill_null(float('inf')))
        .with_columns(pl.lit('InterProScan').alias('Source'))
        .with_columns(pl.lit(None).alias('Preferred Name'))
        .select(['ID', 'Accession', 'Description', 'Score', 'Source', 'Preferred Name'])
    )
    genes = filter_interpro_df['ID'].unique()
    logger.info(f'InterProScan过滤后记录共{filter_interpro_df.shape[0]}条，{len(genes)}个基因')
    del interpro_df

    # eggNOG-mapper 结果读取
    eggnog_df = pl.read_csv(
        eggnog_file, separator='\t', comment_prefix='##', has_header=True, null_values='-'
    )
    logger.info(f'eggNOG-mapper数据读取完毕，共{eggnog_df.shape[0]}条')

    filter_eggnog_df = (
        eggnog_df.select(['#query', 'evalue', 'Description', 'Preferred_name', 'KEGG_ko'])
        .with_columns(
            pl.col('#query').str.slice(0, pl.col('#query').str.len_chars() - 2).alias('ID')
        )
        .rename(
            {
                'evalue': 'Score',
                'Preferred_name': 'Preferred Name',
                'KEGG_ko': 'Accession',
            }
        )
        .filter(pl.col('Description').is_not_null())
        .unique()
        .with_columns(pl.lit('eggNOG-mapper').alias('Source'))
        .select(['ID', 'Accession', 'Description', 'Score', 'Source', 'Preferred Name'])
    )
    genes = filter_eggnog_df['ID'].unique()
    logger.info(f'eggNOG-mapper过滤后记录共{filter_eggnog_df.shape[0]}条，{len(genes)}个基因')

    # 合并
    filter_df = (
        pl.concat([filter_interpro_df, filter_eggnog_df], how='vertical_relaxed')
        .group_by(['ID', 'Accession', 'Description', 'Source', 'Preferred Name'])
        .agg([pl.col('Score').min().alias('Score')])
        .sort(['ID', 'Score', 'Accession'], descending=False)
    )
    genes = filter_df['ID'].unique()
    logger.info(f'合并后记录共{filter_df.shape[0]}条，{len(genes)}个基因')
    filter_df.write_csv(output_file)

    return filter_df


async def annotate_gene_by_llm(gene_name: str, gene_df: pl.DataFrame) -> dict[str, Any]:
    """使用大模型对单个基因功能进行注释

    Args:
        gene_name: 基因名称
        gene_df: 截取的注释dataframe

    Returns:
        注释结果
    """

    lines = []
    desc_list = []
    for row in gene_df.iter_rows(named=True):
        lines.append(f'{row["Description"]} [{row["Preferred Name"]}] ({row["Source"]})')
        desc_list.append(f'{row["Accession"]}|{row["Description"]}|{row["Source"]}|{row["Score"]}')
    info_text = '\n'.join(lines)

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=SYSTEM_PROMPT), ('human', HUMAN_PROMPT)]
    )
    chain = prompt | llm.with_structured_output(Annotate)
    result: Annotate = await chain.ainvoke({'input': info_text})

    result_dict = {
        'ID': gene_name,
        'Description': f'[{result.name}] - [{result.description}]',
        'All Description': ';'.join(desc_list),
    }

    return result_dict


async def run_analyse(
    gene_df: pl.DataFrame,
    output_file: str,
    concurrency: int = 5,
    top: int = 5,
    cut: Optional[int] = None,
):
    """异步执行注释任务

    Args:
        gene_df: 过滤后的数据表
        output_file: 结果保存文件（TSV）
        concurrency: 最大并发数
        top: 每一条记录取前N条结果询问AI
        cut: 截取多少条记录（测试用）
    """

    genes = gene_df['ID'].unique()
    if cut:
        genes = genes[:cut]

    result = []

    semaphore = asyncio.Semaphore(concurrency)

    async def sem_task(_gene):
        async with semaphore:
            sub_df = gene_df.filter(pl.col('ID') == _gene).top_k(top, by=['Score'])
            return await annotate_gene_by_llm(_gene, sub_df)

    tasks = [sem_task(gene) for gene in genes]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(genes)):
        annotation = await coro
        result.append(annotation)

    output_df = pl.DataFrame(result)
    output_df.write_csv(output_file, separator='\t')


def main():
    origin_df = load_data(
        'report/IMET1v2.tsv', 'report/gene.emapper.annotations', 'report/filtered.csv'
    )
    asyncio.run(run_analyse(origin_df, 'report/result.tsv', cut=5))


if __name__ == '__main__':
    main()
