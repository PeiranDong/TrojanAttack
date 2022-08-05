# NL2SQL

This downstream task refers to Content Enhanced BERT-based Text-to-SQL Generation https://arxiv.org/abs/1910.07179

# Requirements

python 3.6

records 0.5.3   

torch 1.1.0   

# Run

1, Data prepare:
Download all origin data( https://drive.google.com/file/d/1iJvsf38f16el58H4NPINQ7uzal5-V4v4 or https://download.csdn.net/download/guotong1988/13008037) and put them at `data_and_model` directory.

Then run
`data_and_model/output_entity.py`

2, Train and eval:

`train.py`



# Reference 

https://github.com/naver/sqlova
