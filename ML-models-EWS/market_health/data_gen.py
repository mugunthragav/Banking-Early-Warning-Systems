# data_gen.py
from src.data.synthetic_data import generate_synthetic_data

if __name__ == '__main__':
    symbols = ['@CL#C', '@C#C', '@ES', '@NG', '@SI', '@TY', '@AD', '@EC', '@6E', '@GC', '@LE', '@NQ', '@SB', '@YM']
    generate_synthetic_data(symbols, n_rows_min=600000, n_rows_day=2520)