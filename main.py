import polars as pl
from loguru import logger


def load_data(tsv_file: str, output_file: str) -> pl.DataFrame:
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
        .unique()
        .group_by(['ID', 'InterPro annotations accession', 'InterPro annotations description'])
        .agg([pl.col('Score').max().alias('Score')])
        .sort(['ID', 'Score', 'InterPro annotations accession'], descending=True)
    )
    not_null_df.write_csv(output_file)
    genes = not_null_df['ID'].unique()
    logger.info(f'过滤后注释非空记录共{not_null_df.shape[0]}条，{len(genes)}个基因')

    return not_null_df


def annotate_gene_by_llm(gene_df: pl.DataFrame) -> str:
    print(gene_df)


def main():
    origin_df = load_data('report/IMET1v2.tsv', 'report/filtered.csv')
    genes = origin_df['ID'].unique()
    for gene in genes:
        sub_df = origin_df.filter(pl.col('ID') == gene)
        annotate_gene_by_llm(sub_df)
        break


if __name__ == '__main__':
    main()
