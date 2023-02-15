# class TimeFrequencyDecider:
#     def __init__(self, column,
#                  col_name,
#                  time_series_handler: TimeSeriesHandler = None,
#                  data_info: DataInfo = None):
#         self.col_name = col_name  # the name of the date col
#         self.column = column
#         self.time_series_handler = time_series_handler or TimeSeriesHandler()
#         self.data_info = data_info
#         self.df = self.data_info.data
#         self.id_cols = self.time_series_handler.id_cols or []