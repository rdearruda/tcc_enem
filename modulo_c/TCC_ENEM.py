# Databricks notebook source
# MAGIC %md
# MAGIC # Objetivo
# MAGIC 
# MAGIC O objetivo é prever a nota da prova de **Redação** a partir das outras 4.
# MAGIC 
# MAGIC Os dados contidos nessa base são de 2015 a 2019 e serão utilizados os dados de 2015 a 2018 para treinamento/teste. 
# MAGIC 
# MAGIC E o modelo treinado será aplicado para tentar prever a nota do ano de 2019.

# COMMAND ----------

# MAGIC %md
# MAGIC # Libs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando Libs

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
import pandas_profiling as pp
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preparation
# MAGIC 
# MAGIC As tabelas utilizadas para o módulo C foram criadas nos módulos A e B.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Input

# COMMAND ----------

# MAGIC %md
# MAGIC ### Paths dos Arquivos
# MAGIC Aponta para os paths do FileStore para onde os arquivos foram importados

# COMMAND ----------

path_cadastral = '/FileStore/tables/DIM_CADASTRAL_RR.csv'
path_ensino = '/FileStore/tables/DIM_ENSINO_RR.csv'
path_presenca = '/FileStore/tables/DIM_PRESENCA_RR.csv'
path_local = '/FileStore/tables/DIM_LOCAL_PROVA_RR.csv'
path_socio = '/FileStore/tables/DIM_SOCIO_ECONOMICO_RR.csv'
path_municipio = '/FileStore/tables/DIM_IBGE_MUNICIPIO.csv'
path_nota = '/FileStore/tables/FAT_NOTA_RR.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configurando Options

# COMMAND ----------

file_type = 'csv'
infer_schema = 'true'
first_row_is_header = 'true'
delimiter = '\t'
encode = 'ISO-8859-1'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dim_Cadastral

# COMMAND ----------

df_cadastral = spark.read.format(file_type) \
                    .option('inferSchema', infer_schema) \
                    .option('header', first_row_is_header) \
                    .option('sep', delimiter) \
                    .option('encoding', encode) \
                    .load(path_cadastral) \
                    .drop('COD_MUN_NSC', 'COD_MUN_RSD', 'FL_IDS', 'FL_NOM_SCL', 'FL_INS_TRE', 'COD_NAC', 'DSC_NAC', 'DSC_SEX', 'DSC_RCA', 'DSC_EST_CVL')

display(df_cadastral)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dim_Ensino

# COMMAND ----------

df_ensino = spark.read.format(file_type) \
                    .option('inferSchema', infer_schema) \
                    .option('header', first_row_is_header) \
                    .option('sep', delimiter) \
                    .option('encoding', encode) \
                    .load(path_ensino) \
                    .withColumn('TPO_ESC_ENS_MED_DET', F.coalesce(F.col('TPO_ESC_ENS_MED_DET'), F.lit('')))

display(df_ensino)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dim_Presença

# COMMAND ----------

df_presenca = spark.read.format(file_type) \
                    .option('inferSchema', infer_schema) \
                    .option('header', first_row_is_header) \
                    .option('sep', delimiter) \
                    .option('encoding', encode) \
                    .load(path_presenca)

display(df_presenca)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dim_Local_Prova

# COMMAND ----------

df_local = spark.read.format(file_type) \
                    .option('inferSchema', infer_schema) \
                    .option('header', first_row_is_header) \
                    .option('sep', delimiter) \
                    .option('encoding', encode) \
                    .load(path_local)

display(df_local)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dim_Socio_Economica

# COMMAND ----------

df_socio = spark.read.format(file_type) \
                    .option('inferSchema', infer_schema) \
                    .option('header', first_row_is_header) \
                    .option('sep', delimiter) \
                    .option('encoding', encode) \
                    .load(path_socio)

display(df_socio)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dim_IBGE_Municipio

# COMMAND ----------

df_municipio = spark.read.format(file_type) \
                    .option('inferSchema', infer_schema) \
                    .option('header', first_row_is_header) \
                    .option('sep', delimiter) \
                    .option('encoding', encode) \
                    .load(path_municipio)

display(df_municipio)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fat_Notas

# COMMAND ----------

df_nota = spark.read.format(file_type) \
                    .option('inferSchema', infer_schema) \
                    .option('header', first_row_is_header) \
                    .option('sep', delimiter) \
                    .option('encoding', encode) \
                    .load(path_nota) \
                    .withColumn('NOT_CIE_HUM', F.round(F.regexp_replace(F.col('NOT_CIE_HUM'), ',', '.').cast('double'), 2)) \
                    .withColumn('NOT_CIE_NAT', F.round(F.regexp_replace(F.col('NOT_CIE_NAT'), ',', '.').cast('double'), 2)) \
                    .withColumn('NOT_MAT', F.round(F.regexp_replace(F.col('NOT_MAT'), ',', '.').cast('double'), 2)) \
                    .withColumn('NOT_LNG_COD', F.round(F.regexp_replace(F.col('NOT_LNG_COD'), ',', '.').cast('double'), 2)) \
                    .withColumn('NOT_RDC', F.round(F.regexp_replace(F.col('NOT_RDC'), ',', '.').cast('double'), 2)) \
                    .withColumn('NOT_MED', F.round(F.regexp_replace(F.col('NOT_MED'), ',', '.').cast('double'), 2))

display(df_nota)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join
# MAGIC 
# MAGIC Os critérios para selecionar quais inscritos comporão a base a ser utilizada:
# MAGIC - O inscrito precisa ter comparecido em todas as provas
# MAGIC - O inscrito precisa estar na base de notas

# COMMAND ----------

df_output = df_cadastral.alias('a') \
                .join(df_presenca.alias('b'), ((df_cadastral.NUM_INS == df_presenca.NUM_INS) & (df_presenca.DSC_TPO_PSC  == 'Presente')), 'inner') \
                .join(df_nota.alias('c'), df_cadastral.NUM_INS == df_nota.NUM_INS, 'inner') \
                .join(df_ensino.alias('d'), df_cadastral.NUM_INS == df_ensino.NUM_INS, 'left') \
                .join(df_local.alias('e'), df_cadastral.NUM_INS == df_local.NUM_INS, 'left') \
                .join(df_municipio.alias('f'), df_local.COD_MUN_PRV == df_municipio.COD_MUN_IBGE, 'left') \
                .join(df_socio.alias('g'), df_cadastral.NUM_INS == df_socio.NUM_INS, 'left') \
                .select(F.col('a.*') \
                        , F.col('TPO_ESC_ENS_MED_DET') \
                        , F.col('NOM_MUN') \
                        , F.col('UF') \
                        , F.col('REG_UF') \
                        , F.col('Q001') \
                        , F.col('Q002') \
                        , F.col('Q003') \
                        , F.col('Q004') \
                        , F.col('Q006') \
                        , F.col('Q014') \
                        , F.col('NOT_CIE_HUM') \
                        , F.col('NOT_CIE_NAT') \
                        , F.col('NOT_MAT') \
                        , F.col('NOT_LNG_COD') \
                        , F.col('NOT_RDC') \
                        , F.col('NOT_MED') )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output

# COMMAND ----------

display(df_output)

# COMMAND ----------

# MAGIC %md
# MAGIC # Profiling
# MAGIC 
# MAGIC O profiling é possibilita entender como os dados que serão trabalhados estão

# COMMAND ----------

df_input_ml = df_output.toPandas()
profile = pp.ProfileReport(df_input_ml, title='Profiling ENEM', html={'style':{'full_width':True}})

# COMMAND ----------

displayHTML(profile.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # Machine Learning

# COMMAND ----------

# MAGIC %md
# MAGIC ## DataFrame inicial com os dados que serão utilizados para treinar o modelo
# MAGIC Contém os dados de 2015 até 2018

# COMMAND ----------

df_treino_full = df_input_ml.loc[df_input_ml['ANO_ENEM'] <= 2018]

df_treino_full

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criando o modelo

# COMMAND ----------

# MAGIC %md
# MAGIC ### Construindo o dataframe com as notas das provas que serão utilizadas para prever

# COMMAND ----------

nota_treino_input = ['NOT_CIE_HUM', 'NOT_CIE_NAT', 'NOT_LNG_COD', 'NOT_MAT']
df_treino_notas_input = df_treino_full[nota_treino_input]
df_treino_notas_input

# COMMAND ----------

# MAGIC %md
# MAGIC ### Construindo o dataframe com as notas da prova que se quer prever

# COMMAND ----------

nota_treino_output = ['NOT_RDC']
df_treino_nota_output = df_treino_full[nota_treino_output]
df_treino_nota_output

# COMMAND ----------

# MAGIC %md
# MAGIC ### Treinando o Modelo

# COMMAND ----------

#Segmentando as bases de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(df_treino_notas_input, df_treino_nota_output, test_size = 0.25, random_state = 4000)

#Instanciando o modelo
modelo_lin_svr = LinearSVR(random_state = 4000)

#Submetendo os dados de treino ao modelo criado
modelo_lin_svr.fit(x_treino, y_treino)

#Submentendo os dados de teste ao modelo treinado
pred_rdc_arvore = modelo_lin_svr.predict(x_teste)

# COMMAND ----------

# MAGIC %md
# MAGIC Criando o DataFrame com os resultados obtidos

# COMMAND ----------

df_resultado = pd.DataFrame()
df_resultado = x_teste
df_resultado = y_teste
df_resultado['PREVISAO'] = pred_rdc_arvore

df_resultado = spark.createDataFrame(df_resultado) 

df_resultado.createOrReplaceTempView('df_resultado')
display(df_resultado)

# COMMAND ----------

# MAGIC %md
# MAGIC Criando a base com os dados de teste e os dados que serão utilizados para avaliar o modelo

# COMMAND ----------

#Calculado a diferença entre a nota real e a nota prevista
diff_nota = F.col('NOT_RDC') - F.col('PREVISAO')

#Converte em número positivo a diferença da nota para os casos em que a nota prevista for maior que a real
diff_modulo = F.when( diff_nota < 0, diff_nota*(-1)) \
               .otherwise(diff_nota)

#Cria um Flag Informando se previu uma nota maior, menor ou igual a real
dif_positiva = F.when( diff_nota < 0, 'Atribuiu Nota Maior') \
               .when( diff_nota > 0, 'Atribuiu Nota Menor') \
               .otherwise('Atribuiu Nota Exata')

#Calcula o percentual entre a nota prevista e a nota real
perc_diff = F.when( diff_nota < 0, ( ( (F.col('PREVISAO')/F.col('NOT_RDC')) *100) -100) ) \
             .when( diff_nota > 0, (100 - ((F.col('PREVISAO')/F.col('NOT_RDC'))*100) )) \
             .otherwise(0)

#Cria as faixas percentuais que serão utilizadas para avaliar o modelo
faixa_dif = F.when( F.col('Percentual_da_Diferenca').between(0, 10), '0, 10') \
              .when( F.col('Percentual_da_Diferenca').between(11, 20), '11, 20') \
              .when( F.col('Percentual_da_Diferenca').between(21, 30), '21, 30') \
              .when( F.col('Percentual_da_Diferenca').between(31, 40), '31, 40') \
              .when( F.col('Percentual_da_Diferenca').between(41, 50), '41, 50') \
              .when( F.col('Percentual_da_Diferenca').between(51, 60), '51, 60') \
              .when( F.col('Percentual_da_Diferenca').between(61, 70), '61, 70') \
              .when( F.col('Percentual_da_Diferenca').between(71, 80), '71, 80') \
              .when( F.col('Percentual_da_Diferenca').between(81, 90), '81, 90') \
              .when( F.col('Percentual_da_Diferenca').between(91, 100), '91, 100') \
              .when( F.col('Percentual_da_Diferenca') > 100, '> 100') \
              .otherwise(None)

#DataFrame com os campos finais do teste realizado
df_resultado_spark = df_resultado.withColumn('PREVISAO', F.col('PREVISAO').cast('int')) \
                                  .withColumn('Diferenca', diff_modulo.cast('int')) \
                                  .withColumn('Tipo_da_Diferenca', dif_positiva) \
                                  .withColumn('Percentual_da_Diferenca', perc_diff.cast('int')) \
                                  .withColumn('Faixa_da_Diferenca', faixa_dif) \
                                  .withColumnRenamed('NOT_RDC', 'Nota_Real') \
                                  .withColumnRenamed('PREVISAO', 'Nota_Prevista') \
                                  .select(F.col('Nota_Real') \
                                         , F.col('Nota_Prevista') \
                                         , F.col('Diferenca') \
                                         , F.col('Tipo_da_Diferenca') \
                                         , F.col('Percentual_da_Diferenca') \
                                         , F.col('Faixa_da_Diferenca'))

df_resultado_spark.createOrReplaceTempView('df_resultado_spark')
display(df_resultado_spark)

# COMMAND ----------

# MAGIC %md
# MAGIC Considerando os critérios adotados a perfomance obtida pelo modelo foi de **81,41%**.
# MAGIC 
# MAGIC 
# MAGIC Ou seja, para **81,41%** dos inscritos o modelo atribuiu uma nota dentro do intervalo considerando bom (até 30%, acima ou abaixo, da nota real)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select (select count(1) from df_resultado_spark where Percentual_da_Diferenca <= 30) / (select count(1) from df_resultado_spark) *100 as percentual_acerto

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aplicando o Modelo

# COMMAND ----------

# MAGIC %md
# MAGIC ### DataFrame Previsão com os dados que serão utilizados para treinar o modelo
# MAGIC Contém os dados do ano 2019

# COMMAND ----------

df_prever_full = df_input_ml.loc[df_input_ml['ANO_ENEM'] == 2019]

df_prever_full

# COMMAND ----------

# MAGIC %md
# MAGIC ### Construindo o dataframe com as notas das provas que serão utilizadas para prever

# COMMAND ----------

nota_prever_input = ['NOT_CIE_HUM', 'NOT_CIE_NAT', 'NOT_LNG_COD', 'NOT_MAT']
df_prever_notas_input = df_prever_full[nota_prever_input]
df_prever_notas_input

# COMMAND ----------

# MAGIC %md
# MAGIC ### Construindo o dataframe com as notas da prova que se quer prever

# COMMAND ----------

nota_prever_output = ['NOT_RDC']
df_prever_nota_output = df_prever_full[nota_prever_output]
df_prever_nota_output

# COMMAND ----------

# MAGIC %md
# MAGIC Aplicando o modelo

# COMMAND ----------

df_previsao = modelo_lin_svr.predict(df_prever_notas_input)

# COMMAND ----------

# MAGIC %md
# MAGIC Criando o DataFrame com os resultados obtidos

# COMMAND ----------

df_resultado_previsao = pd.DataFrame()
df_resultado_previsao = df_prever_notas_input
df_resultado_previsao = df_prever_nota_output
df_resultado_previsao['PREVISAO'] = df_previsao

df_resultado_previsao = spark.createDataFrame(df_resultado_previsao) 

df_resultado_previsao.createOrReplaceTempView('df_resultado_previsao')

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from df_resultado_previsao

# COMMAND ----------

# MAGIC %md
# MAGIC Criando a base com os inputados e os dados que serão utilizados para avaliar o modelo

# COMMAND ----------

#Calculado a diferença entre a nota real e a nota prevista
diff_nota = F.col('NOT_RDC') - F.col('PREVISAO')

#Converte em número positivo a diferença da nota para os casos em que a nota prevista for maior que a real
diff_modulo = F.when( diff_nota < 0, diff_nota*(-1)) \
               .otherwise(diff_nota)

#Cria um Flag Informando se previu uma nota maior, menor ou igual a real
dif_positiva = F.when( diff_nota < 0, 'Atribuiu Nota Maior') \
               .when( diff_nota > 0, 'Atribuiu Nota Menor') \
               .otherwise('Atribuiu Nota Exata')

#Calcula o percentual entre a nota prevista e a nota real
perc_diff = F.when( diff_nota < 0, ( ( (F.col('PREVISAO')/F.col('NOT_RDC')) *100) -100) ) \
             .when( diff_nota > 0, (100 - ((F.col('PREVISAO')/F.col('NOT_RDC'))*100) )) \
             .otherwise(0)

#Cria as faixas percentuais que serão utilizadas para avaliar o modelo
faixa_dif = F.when( F.col('Percentual_da_Diferenca').between(0, 10), '0, 10') \
              .when( F.col('Percentual_da_Diferenca').between(11, 20), '11, 20') \
              .when( F.col('Percentual_da_Diferenca').between(21, 30), '21, 30') \
              .when( F.col('Percentual_da_Diferenca').between(31, 40), '31, 40') \
              .when( F.col('Percentual_da_Diferenca').between(41, 50), '41, 50') \
              .when( F.col('Percentual_da_Diferenca').between(51, 60), '51, 60') \
              .when( F.col('Percentual_da_Diferenca').between(61, 70), '61, 70') \
              .when( F.col('Percentual_da_Diferenca').between(71, 80), '71, 80') \
              .when( F.col('Percentual_da_Diferenca').between(81, 90), '81, 90') \
              .when( F.col('Percentual_da_Diferenca').between(91, 100), '91, 100') \
              .when( F.col('Percentual_da_Diferenca') > 100, '> 100') \
              .otherwise(None)

#DataFrame com os campos finais do teste realizado
df_resultado_previsao_spark = df_resultado_previsao.withColumn('PREVISAO', F.col('PREVISAO').cast('int')) \
                                                    .withColumn('Diferenca', diff_modulo.cast('int')) \
                                                    .withColumn('Tipo_da_Diferenca', dif_positiva) \
                                                    .withColumn('Percentual_da_Diferenca', perc_diff.cast('int')) \
                                                    .withColumn('Faixa_da_Diferenca', faixa_dif) \
                                                    .withColumnRenamed('NOT_RDC', 'Nota_Real') \
                                                    .withColumnRenamed('PREVISAO', 'Nota_Prevista') \
                                                    .select(F.col('Nota_Real') \
                                                           , F.col('Nota_Prevista') \
                                                           , F.col('Diferenca') \
                                                           , F.col('Tipo_da_Diferenca') \
                                                           , F.col('Percentual_da_Diferenca') \
                                                           , F.col('Faixa_da_Diferenca'))

df_resultado_previsao_spark.createOrReplaceTempView('df_resultado_previsao_spark')
display( df_resultado_previsao_spark)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select (select count(1) from df_resultado_previsao_spark where Percentual_da_Diferenca <= 30) / (select count(1) from df_resultado_previsao_spark) *100 as percentual_acerto

# COMMAND ----------


