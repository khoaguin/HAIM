import pandas as pd
from glob import glob
from tqdm import tqdm

# Potentially could supply los-result or mortality-result
folder_name = ''
# Potentially could supply los, mortality
task = ''

lis = []
for i in glob(folder_name + '/*'):
    lis.append(i.split('-')[0].split('/')[1])
    
data_modality_lis = []
aucroc_mean_lis = []
aucpr_mean_lis = []
f1_mean_lis = []
aucroc_std_lis = []
aucpr_std_lis = []
f1_std_lis = []

for i in tqdm(range(2047)):
    auc_lis = []
    f1_lis = []
    data_modality = ''
    for j in range(5):
        df = pd.read_csv(folder_name + '/{}-{}.csv'.format(i, j))
        data_modality = df['Data Modality'].values[0]
        aucroc_lis.append(df['Test AUCROC'].values[0])
        aucpr_lis.append(df['Test AUCPR'].values[0])
        f1_lis.append(df['Test F1 Score'].values[0])

    aucroc_mean_lis.append(np.mean(aucroc_lis))
    aucpr_mean_lis.append(np.mean(aucpr_lis))
    f1_mean_lis.append(np.mean(f1_lis))

    aucroc_std_lis.append(np.std(aucroc_lis))
    aucpr_std_lis.append(np.std(aucpr_lis))
    f1_std_lis.append(np.std(f1_lis))

    data_modality_lis.append(data_modality)
    
df_result = pd.DataFrame(columns = ['model_names', 'AUC_mean'])
df_result['model_names'] = data_modality_lis
df_result['AUCROC_mean'] = aucroc_mean_lis
df_result['AUCROC_std'] = aucroc_std_lis
df_result['AUCPR_mean'] = aucpr_mean_lis
df_result['AUCPR_std'] = aucpr_std_lis
df_result['F1_mean'] = f1_mean_lis
df_result['F1_std'] = f1_std_lis

df_result['model_names_cleaned'] = [i[1:-1] for i in df_result['model_names']]
df_result['Sources'] = [i.replace("'", '').split(', ') for i in df_result['model_names_cleaned']]
df_result['Sources'][0:10] = [[i[0][:-1]] for i in df_result['Sources'][0:10]]

d = {'viz': ['vp', 'vd', 'vmp', 'vmd'], 'tab': ['demo'], 'txt':['n_ech', 'n_ecg'], 'ts':['ts_pe', 'ts_ce', 'ts_le']}

def build_moda(df, moda):
    viz = np.zeros(len(df))
    for i in range(len(df)):
        for j in range(len(df['Sources'][i])):
            viz[i] = viz[i] or (df['Sources'][i][j] in d[moda])
    return viz


df_result['viz'] = build_moda(df_result, 'viz')
df_result['txt'] = build_moda(df_result, 'txt')
df_result['tab'] = build_moda(df_result, 'tab')
df_result['ts'] = build_moda(df_result, 'ts')
df_result['Number of Modalities'] = df_result['viz'] + df_result['txt'] + df_result['tab'] + df_result['ts']
df_result['Number of Sources'] = list(map(len, df_result['model_names_cleaned'].str.split(', ')))

df_result = df_result.drop(['model_names_cleaned', 'model_names'], axis = 1)

df_result.to_csv("AUC_All_los_pred.csv")
df_result[['AUCROC_mean']].to_csv('result-table/AUCROC_mean_{}_pred.csv'.format(task))
df_result[['AUCPR_mean']].to_csv('result-table/AUCPR_mean_{}_pred.csv'.format(task)
df_result[['AUCROC_std']].to_csv('result-table/AUCROC_std_{}_pred.csv'.format(task)
df_result[['AUCPR_std']].to_csv('result-table/AUCPR_std_{}_pred.csv'.format(task)
df_result[['F1_mean']].to_csv('result-table/F1_mean_{}_pred.csv'.format(task)
df_result[['F1_std']].to_csv('result-table/F1_std_{}_pred.csv'.format(task)