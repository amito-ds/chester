# import pandas as pd
#
# from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
# from run_full_cycle import run_tcap, DataSpec
#
# df1 = load_data_pirates().assign(target='chat_logs')
# df2 = load_data_king_arthur().assign(target='pirates')
# df = pd.concat([df1, df2])
#
# out = run_tcap(
#     data_spec=DataSpec(df=df, text_column='text', target_column='target'),
# )