import os
import pandas as pd

# plot_dir = [
#     "./PDformer_two_step_spy",
#     "./PDformer_two_step",
#     "./PDformer_pu_bagging",
#     "./PDformer_all",
# ]

start = 195
end = start + 5

# file = pd.read_excel("DMGAT.xlsx", index_col=0, sheet_name=None)
# fw = pd.ExcelWriter(rf"DMGAT_result_compare{start}.xlsx")
# file = pd.read_excel("DMGAT_1hot.xlsx", index_col=0, sheet_name=None)
# fw = pd.ExcelWriter(rf"DMGAT_1hot_result_compare{start}.xlsx")
# file = pd.read_excel("DMGAT_solid_valid.xlsx", index_col=0, sheet_name=None)
# fw = pd.ExcelWriter(rf"DMGAT_solid_valid_result_compare{start}.xlsx")
# file = pd.read_excel("DMGAT_tfidf.xlsx", index_col=0, sheet_name=None)
# fw = pd.ExcelWriter(rf"DMGAT_tfidf_result_compare{start}.xlsx")
file = pd.read_excel("DMGAT_bow.xlsx", index_col=0, sheet_name=None)
fw = pd.ExcelWriter(rf"DMGAT_bow_result_compare{start}.xlsx")
# file = pd.read_excel("DMGAT_2_gcn_4_gat.xlsx", index_col=0, sheet_name=None)
# fw = pd.ExcelWriter(rf"DMGAT_2_gcn_4_gat_result_compare{start}.xlsx")
# file = pd.read_excel("DMGAT_noDM.xlsx", index_col=0, sheet_name=None)
# fw = pd.ExcelWriter(rf"DMGAT_noDM_result_compare{start}.xlsx")
# file = pd.read_excel("DMGAT_noGCN.xlsx", index_col=0, sheet_name=None)
# fw = pd.ExcelWriter(rf"DMGAT_noGCN_result_compare{start}.xlsx")
# file = pd.read_excel("DMGAT_noGAT.xlsx", index_col=0, sheet_name=None)
# fw = pd.ExcelWriter(rf"DMGAT_noGAT_result_compare{start}.xlsx")
# file = pd.read_excel("DMGAT_lnc.xlsx", index_col=0, sheet_name=None)
# fw = pd.ExcelWriter(rf"DMGAT_lnc_result_compare{start}.xlsx")
# file = pd.read_excel("DMGAT_mi.xlsx", index_col=0, sheet_name=None)
# fw = pd.ExcelWriter(rf"DMGAT_mi_result_compare{start}.xlsx")

n = 0
sheet0 = file["fold0"][start:end]
sheet1 = file["fold1"][start:end]
sheet2 = file["fold2"][start:end]
sheet3 = file["fold3"][start:end]
sheet4 = file["fold4"][start:end]
sheet = pd.concat((sheet0, sheet1, sheet2, sheet3, sheet4))
sheet_mean = sheet.mean()
sheet_std = sheet.std()
sheet_comb = pd.concat([sheet, sheet_mean.to_frame().T], ignore_index=True)
sheet_comb = pd.concat([sheet_comb, sheet_std.to_frame().T], ignore_index=True)

result = sheet_mean.round(4).astype(str) + "±" + sheet_std.round(3).astype(str)
result = result.to_frame().T.rename(index={0: 'DMGAT'})

result.to_excel(fw, sheet_name="all_result")
print(result[['auc', 'aupr']])
sheet_comb.to_excel(fw, sheet_name='DMGAT')

fw.close()
