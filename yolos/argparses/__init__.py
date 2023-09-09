import argparse
from cgitb import text
import textwrap

__all__ = [
    "argumentParse"
]

class ArgsHelpFormat(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter
):
    pass

initParses = textwrap.dedent(
    """
    初期化 : 環境立ち上げ
    /Task Project
        |- Datasets (学習・評価用データ格納領域)
        |   |-/TrainData
        |   |-/TestData
        |- Outputs (学習進捗，学習済みパラメータ，学習データ出力，etc...)
    """
)

def argumentParse():
    
    parses = argparse.ArgumentParser(
        description='',
        formatter_class=ArgsHelpFormat,
        epilog=
    )


if __name__ == "___main__":
    pass