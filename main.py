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
SYSTEM_PROMPT = """用户会给出基因的 InterPro 注释。请用20字以内的中文总结该基因的主要功能（若无可靠信息写“未知功能”）"""

llm = ChatOpenAI(model='o4-mini', base_url='https://aihubmix.com/v1', api_key=API_KEY)


class Annotate(BaseModel):
    description: str = Field(description='基因的主要功能')


def load_data(tsv_file: str, output_file: str) -> pl.DataFrame:
    """加载数据并过滤

    Args:
        tsv_file: 原始输入（TSV）
        output_file: 用于暂存中间结果的文件（CSV）

    Returns:
        过滤后的dataframe
    """
    df = pl.read_csv(
        tsv_file,
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
    logger.info(f'数据读取完毕，共{df.shape[0]}条')

    not_null_df = (
        df.select(
            [
                'ID',
                'Score',
                'InterPro annotations accession',
                'InterPro annotations description',
            ]
        )
        .filter(pl.col('InterPro annotations description').is_not_null())
        .with_columns(pl.col('Score').fill_null(0))
        .unique()
        .group_by(['ID', 'InterPro annotations accession', 'InterPro annotations description'])
        .agg([pl.col('Score').max().alias('Score')])
        .sort(['ID', 'Score', 'InterPro annotations accession'], descending=True)
    )
    not_null_df.write_csv(output_file)
    genes = not_null_df['ID'].unique()
    logger.info(f'过滤后注释非空记录共{not_null_df.shape[0]}条，{len(genes)}个基因')

    return not_null_df


async def annotate_gene_by_llm(gene_name: str, gene_df: pl.DataFrame) -> dict[str, Any]:
    """使用大模型对单个基因功能进行注释

    Args:
        gene_name: 基因名称
        gene_df: 截取的注释dataframe

    Returns:
        注释结果
    """

    lines = ['| InterPro Accession | InterPro Description | Score |', '| ---- | ---- | ---- |']
    desc_list = []
    for row in gene_df.iter_rows():
        accession, description, score = row
        lines.append(f'| {accession} | {description} | {score} |')
        desc_list.append(f'{accession},{description},{score}')
    info_text = '\n'.join(lines)

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=SYSTEM_PROMPT), ('human', '{input}')]
    )
    chain = prompt | llm.with_structured_output(Annotate)
    result: Annotate = await chain.ainvoke({'input': info_text})

    result_dict = {
        'ID': gene_name,
        'Description': result.description,
        'All Description': ';'.join(desc_list),
    }

    return result_dict


async def run_analyse(
    gene_df: pl.DataFrame, output_file: str, concurrency: int = 5, cut: Optional[int] = None
):
    """异步执行注释任务

    Args:
        gene_df: 过滤后的数据表
        output_file: 结果保存文件（TSV）
        concurrency: 最大并发数
        cut: 截取
    """

    genes = gene_df['ID'].unique()
    if cut:
        genes = genes[:cut]

    result = []

    semaphore = asyncio.Semaphore(concurrency)

    async def sem_task(_gene):
        async with semaphore:
            sub_df = gene_df.filter(pl.col('ID') == _gene).drop(['ID'])
            return await annotate_gene_by_llm(_gene, sub_df)

    tasks = [sem_task(gene) for gene in genes]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(genes)):
        annotation = await coro
        result.append(annotation)

    output_df = pl.DataFrame(result)
    output_df.write_csv(output_file, separator='\t')


def main():
    origin_df = load_data('report/IMET1v2.tsv', 'report/filtered.csv')
    asyncio.run(run_analyse(origin_df, 'report/result.tsv'))


if __name__ == '__main__':
    main()
