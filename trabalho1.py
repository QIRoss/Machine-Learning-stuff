# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Imports

# %%
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import LinearSVC

# %% [markdown]
# ### Desafio:
# 
# Este é o Trabalho 1 de avaliação da disciplina EEL891 (Introdução ao Aprendizado de Máquina) para a turma do período 2020-1.
# 
# Neste trabalho você construirá um classificador para apoio à decisão de aprovação de crédito.
# 
# A ideia é identificar, dentre os clientes que solicitam um produto de crédito (como um cartão de crédito ou um empréstimo pessoal, por exemplo) e que cumprem os pré-requisitos essenciais para a aprovação do crédito, aqueles que apresentem alto risco de não conseguirem honrar o pagamento, tornando-se inadimplentes.
# 
# Para isso, você receberá um arquivo com dados históricos de 20.000 solicitações de produtos de créditos que foram aprovadas pela instituição, acompanhadas do respectivo desfecho, ou seja, acompanhadas da indicação de quais desses solicitantes conseguiram honrar os pagamentos e quais ficaram inadimplentes.
# 
# Com base nesses dados históricos, você deverá construir um classificador que, a partir dos dados de uma nova solicitação de crédito, tente predizer se este solicitante será um bom ou mau pagador.
# 
# O objetivo da competição é disputar com seus colegas quem consegue obter a acurácia mais alta em um conjunto de 5.000 solicitações de crédito aprovadas (diferentes das 20.000 anteriores) cujos desfechos (quitação da dívida ou inadimplência) são mantidos ocultos no site do Kaggle, que medirá automaticamente a taxa de acerto das previsões enviadas pelos competidores, sem revelar o "gabarito".
# %% [markdown]
# ### Leitura de dados

# %%
df_from_eda = pd.read_csv('dataset_from_eda_2021-05-02.csv')
print(df_from_eda.columns)

test = df_from_eda[df_from_eda['origem'] == 'teste']
train = df_from_eda[df_from_eda['origem'] == 'treino']


print(test.shape)
print(train.shape)
# print('\n')
# print(f'test columns {test.columns}')
# print('\n')
# print(f'train columns {train.columns}')
# print('\n')
# print(list(set(train.columns.tolist()) - set(test.columns.tolist())))


# %%



# %%
# test = pd.read_csv('data/desafio 1/conjunto_de_teste.csv')
# train = pd.read_csv('data/desafio 1/conjunto_de_treinamento.csv')

# print(test.shape)
# print(train.shape)
# print('\n')
# print(f'test columns {test.columns}')
# print('\n')
# print(f'train columns {train.columns}')
# print('\n')
# print(list(set(train.columns.tolist()) - set(test.columns.tolist())))


# %%
test.isna().sum() 

# %% [markdown]
# ### Data Cleaning

# %%
# Transform text columns
def encode_string_col(df, col):
    df[col] = df[col].astype('category').cat.codes
    
# sexo 
df_from_eda['sexo'].fillna('N')
encode_string_col(df_from_eda, 'sexo')

# estado_onde_nasceu ... etc (general string columns)
string_cols = ['estado_onde_nasceu', 'estado_onde_reside', 'possui_telefone_residencial',
               'regiao_onde_reside', 'regiao_onde_nasceu', 'idade_bin']

for col in string_cols:
    encode_string_col(df_from_eda, col)

# train test split 

train = df_from_eda[df_from_eda['origem'] == 'treino']
test = df_from_eda[df_from_eda['origem'] == 'teste']

# dropar variável alvo do teste e origem

test.drop(columns=['inadimplente', 'origem'], inplace=True)
train.drop(columns='origem', inplace=True)

# fill nan

test.fillna(0, inplace=True)
train.fillna(0, inplace=True)

print(test.shape)
print(train.shape)

# %% [markdown]
# ### Cleaned test

# %%
test


# %%
test.drop(columns=[
    'estado_onde_nasceu', 'regiao_onde_nasceu', 'estado_onde_reside', 'idade', 
    'dia_vencimento', 'produto_solicitado'
],
           inplace=True)


# %%
test.groupby('ocupacao').size()

# %% [markdown]
# ### Cleaned train

# %%
train


# %%
train.drop(columns=[
    'estado_onde_nasceu', 'regiao_onde_nasceu', 'estado_onde_reside', 'idade', 
    'dia_vencimento', 'produto_solicitado'
],
           inplace=True)


# %%
train.groupby('inadimplente').size()

# %% [markdown]
# #### Split axes

# %%
y_train = train['inadimplente'].to_numpy()
test_ids = test['id_solicitante']
print(set(y_train))
X_train = train.drop(columns=['inadimplente', 'id_solicitante']).select_dtypes(include='number').to_numpy()
X_test = test.drop(columns=['id_solicitante']).select_dtypes(include='number').to_numpy()
print(f'X_test len: {len(X_test)}')


print(X_train.shape)
print(y_train.shape)

# knn = KNeighborsClassifier(n_neighbors=100)

# # # Fit the classifier to the training data
# knn.fit(X_train, y_train)

# pred = knn.predict(X_test)
# print(f'pred len {len(pred)}')
# print(set(pred))
# print(type(pred))

# prediciton_df = pd.DataFrame({'id_solicitante' : test['id_solicitante'].to_numpy(),
#                               'inadimplente' : pred})
# # prediciton_df['inadimplente'] = pd.Series(pred)

# prediciton_df.to_csv(f'data/desafio 1/prediction_knn_100.csv')

# %% [markdown]
# ### KNN

# %%
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
knn.score(X_train, y_train)


# %%
prediciton_df = pd.DataFrame({'id_solicitante' : test['id_solicitante'].to_numpy(),
                          'inadimplente' : pred})

prediciton_df['inadimplente'] = prediciton_df['inadimplente'].astype(int)

prediciton_df.groupby('inadimplente').size()


# %%
prediciton_df.to_csv(f'prediction_knn_8.csv', index=False)


# %%
prediciton_df.groupby('inadimplente').size()

# %% [markdown]
# ### Logistic Regression

# %%
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
prediciton_lr_df = pd.DataFrame({'id_solicitante' : test['id_solicitante'].to_numpy(),
                              'inadimplente' : pred_lr})

lr.score(X_train, y_train)


# %%
prediciton_lr_df['inadimplente'] = prediciton_lr_df['inadimplente'].astype(int)


# %%
prediciton_lr_df.groupby('inadimplente').size()


# %%
prediciton_lr_df.to_csv(f'prediction_lr_7.csv', index=False)

# %% [markdown]
# ### Decision Tree

# %%
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6)
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)
prediciton_dtc_df = pd.DataFrame({'id_solicitante' : test['id_solicitante'].to_numpy(),
                              'inadimplente' : dtc_pred})

dtc.score(X_train, y_train)


# %%
dtc.get_depth()


# %%
prediciton_dtc_df['inadimplente'] = prediciton_dtc_df['inadimplente'].astype(int)


# %%
prediciton_dtc_df.groupby('inadimplente').size()


# %%
prediciton_dtc_df.to_csv(f'prediction_dtc_24.csv', index=False)

# %% [markdown]
# ### SVM

# %%
clf = LinearSVC()
clf.fit(scaled_x_train, y_train)
pred = clf.predict(scaled_x_test)
clf.score(scaled_x_train, y_train)


# %%
set(pred)


# %%
prediction_svm_df = pd.DataFrame({'id_solicitante' : test['id_solicitante'].to_numpy(),
                              'inadimplente' : pred})


# %%
prediction_svm_df


# %%
prediction_svm_df['inadimplente'] = prediction_svm_df['inadimplente'].astype(int)


# %%
prediction_svm_df.groupby('inadimplente').size()


# %%
prediction_svm_df.to_csv(f'prediction_svm_1.csv', index=False)

# %% [markdown]
# ### SGDClassifier

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
scaled_x_train = scaler.transform(X_train)
scaler.fit(X_test)
scaled_x_test = scaler.transform(X_test)


# %%
clf = SGDClassifier()
clf.fit(scaled_x_train, y_train)
pred = clf.predict(scaled_x_test)

clf.score(scaled_x_train, y_train)


# %%
predict_sgd_df = pd.DataFrame({'id_solicitante' : test['id_solicitante'].to_numpy(),
                              'inadimplente' : pred})


# %%
predict_sgd_df['inadimplente'] = predict_sgd_df['inadimplente'].astype(int)


# %%
predict_sgd_df.groupby('inadimplente').size()


# %%
predict_sgd_df.to_csv(f'prediction_sgd_3.csv', index=False)


