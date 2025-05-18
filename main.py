from GS_tools import Gaussian



h=Gaussian()
h.write_to_excel()



'''
# 获取源工作表
source_sheet = wb['Sheet1']
# 复制工作表
new_sheet = wb.copy_worksheet(source_sheet)
new_sheet.title = f'Sheet1_Copy_{timestamp}'
'''