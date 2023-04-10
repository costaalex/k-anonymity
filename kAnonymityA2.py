import datetime

import pandas as pd
import matplotlib.pylab as pl
import matplotlib.patches as patches

'''
 La tupla names contiene i nomi delle colonne del DataFrame che verrà creato leggendo i dati dal file CSV. 
'''
names = (
    'record_id',
    'entity_id',
    'Source',
    'Name',
    'Surname',
    'birth_date',
    'Sex',
    'Place Of Birth',
    'Nationality',
    'Codice Belfiore',
    'age',
)
'''
 La tupla categorical contiene i nomi delle colonne che verranno convertite in tipo di dati categorico.
'''
categorical = {'Source',
               'Name',
               'birth_date',
               'record_id',
               'Place Of Birth',
               'Codice Belfiore'}


'''
calculate_age calcola l’età di un individuo sulla base della sua data di nascita. 
La funzione accetta come argomento una stringa che rappresenta la data di nascita nel formato '%d/%m/%Y' e restituisce un intero che rappresenta l’età dell’individuo.
La funzione inizia ottenendo la data odierna utilizzando il metodo today() della classe date del modulo datetime. 
Successivamente, la data di nascita viene convertita in un oggetto datetime utilizzando il metodo strptime() della classe datetime del modulo datetime.
L’età viene quindi calcolata sottraendo l’anno di nascita dall’anno corrente e sottraendo 1 se il mese 
e il giorno correnti sono minori del mese e del giorno di nascita. Infine, l’età calcolata viene restituita.
'''
def calculate_age(birth_date):
    today = datetime.date.today()
    birth_date = datetime.datetime.strptime(birth_date, '%d/%m/%Y')
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age


'''
lettura dei dati da un file CSV utilizzando la libreria pandas. 
I dati vengono letti in un DataFrame e vengono eseguite alcune operazioni di preprocessing sui dati. 
La funzione calculate_age viene applicata alla colonna birth_date per creare una nuova colonna chiamata age. 
I valori nelle colonne Surname e Nationality vengono trasformati convertendo i primi due caratteri di ogni valore nei loro codici ASCII e concatenandoli. 
I valori risultanti vengono quindi convertiti in interi. Anche i valori nella colonna entity_id vengono preprocessati rimuovendo il prefisso 'entity_' e convertendo i valori risultanti in interi. 
I valori nella colonna Sex vengono sostituiti con codici numerici (1 per 'M' e 2 per 'F') e convertiti in interi. 
Infine, le colonne specificate nel set categorical vengono convertite in tipo di dati categorico.
'''

df = pd.read_csv("./data/k-anonymity/Source_A.csv", sep=",", header=0, names=names[:-1],  index_col=False, engine='python')
df['age'] = df['birth_date'].apply(calculate_age)
n = 2
df['Surname'] = df['Surname'].apply(lambda x: ''.join(str(ord(c)) for c in x[:n].upper()))
df['Surname'] = df['Surname'].astype(int)
df['Nationality'] = df['Nationality'].apply(lambda x: ''.join(str(ord(c)) for c in x[:n].upper()))
df['Nationality'] = df['Nationality'].astype(int)
df['entity_id'] = df['entity_id'].str.replace('entity_', '')
df['entity_id'] = df['entity_id'].astype(int)

df['Sex'] = df['Sex'].str.replace('M', '1')
df['Sex'] = df['Sex'].str.replace('F', '2')
df['Sex'] = df['Sex'].apply(lambda x: int(x))
print(df.head())

for name in categorical:
    df[name] = df[name].astype('category')


'''
get_spans calcola gli intervalli di ciascuna colonna in un DataFrame. 
La funzione accetta come argomenti un DataFrame df, un indice di partizione partition e un dizionario opzionale scale 
che specifica i fattori di scala per ciascuna colonna. La funzione restituisce un dizionario che mappa i nomi delle colonne ai rispettivi intervalli.

La funzione inizia creando un dizionario vuoto spans. Successivamente, per ogni colonna nel DataFrame, 
viene calcolato l’intervallo della colonna. Se la colonna è presente nel set categorical, l’intervallo viene calcolato 
come il numero di valori unici nella partizione specificata. Altrimenti, l’intervallo viene calcolato come la differenza 
tra il valore massimo e il valore minimo nella partizione specificata. Se il dizionario scale è stato fornito, l’intervallo
viene diviso per il fattore di scala corrispondente. 
Infine, l’intervallo calcolato viene aggiunto al dizionario spans e il dizionario viene restituito.
'''
def get_spans(df, partition, scale=None):
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max()-df[column][partition].min()
        if scale is not None:
            span = span/scale[column]
        spans[column] = span
    return spans

full_spans = get_spans(df, df.index)
print(full_spans)


'''
split divide una partizione di un DataFrame in base ai valori di una colonna specificata.
La funzione accetta come argomenti un DataFrame df, un indice di partizione partition e il nome di una colonna column. 
Restituisce una tupla di due indici che rappresentano le due partizioni risultanti.

La funzione inizia creando una variabile dfp che contiene i valori della colonna specificata nella partizione specificata. 
Se la colonna è presente nel set categorical, i valori unici nella partizione vengono divisi in due insiemi di circa uguale dimensione. 
Gli indici dei valori che appartengono al primo insieme vengono restituiti come prima partizione e gli indici dei 
valori che appartengono al secondo insieme vengono restituiti come seconda partizione. 
Altrimenti, viene calcolato il valore mediano nella partizione e gli indici dei valori inferiori al mediano vengono restituiti come prima partizione 
e gli indici dei valori maggiori o uguali al mediano vengono restituiti come seconda partizione.
'''
def split(df, partition, column):
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)

'''
la funzione is_k_anonymous verifica se una partizione di un DataFrame soddisfa il criterio di k-anonimato rispetto a una colonna sensibile specificata. 
Accetta come argomenti un DataFrame df, un indice di partizione partition, il nome di una colonna sensibile sensitive_column e un intero opzionale k che specifica il valore di k da utilizzare (il valore predefinito è 6). 
Restituisce un valore booleano che indica se la partizione soddisfa il criterio di k-anonimato.

La funzione verifica se la lunghezza della partizione è inferiore a k. In tal caso, la partizione non soddisfa il criterio di k-anonimato e la funzione 
restituisce False. Altrimenti, la partizione soddisfa il criterio di k-anonimato e la funzione restituisce True.
'''

def is_k_anonymous(df, partition, sensitive_column, k=6):
    if len(partition) < k:
        return False
    return True


'''
partition_dataset che partiziona un DataFrame in modo che ogni partizione soddisfi un criterio di validità specificato rispetto a una colonna sensibile. 
La funzione accetta come argomenti un DataFrame df, una lista di colonne feature_columns da utilizzare per la partizione, 
il nome di una colonna sensibile sensitive_column, un dizionario scale che specifica i fattori di scala per ciascuna colonna e una funzione is_valid 
che verifica se una partizione soddisfa il criterio di validità. 
Restituisce una lista di indici che rappresentano le partizioni risultanti.
'''
def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid):
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x:-x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions

feature_columns = ['age', 'Surname', 'Nationality']
sensitive_column = 'record_id'
finished_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, is_k_anonymous)

print(len(finished_partitions))


'''
la funzione build_indexes che costruisce un dizionario di indici per le colonne categoriche di un DataFrame. 
La funzione accetta come argomento un DataFrame df e restituisce un dizionario che mappa i nomi delle colonne categoriche ai rispettivi indici.
'''
def build_indexes(df):
    indexes = {}
    for column in categorical:
        values = sorted(df[column].unique())
        indexes[column] = { x : y for x, y in zip(values, range(len(values)))}
    return indexes


'''
La funzione get_coords che calcola le coordinate di una partizione di un DataFrame rispetto a una colonna specificata. 
Accetta come argomenti un DataFrame df, il nome di una colonna column, un indice di partizione partition, 
un dizionario di indici indexes e un valore opzionale offset che specifica l’offset da aggiungere alle coordinate (il valore predefinito è 0.1). 
La funzione restituisce una tupla di due valori che rappresentano le coordinate della partizione rispetto alla colonna specificata.
'''
def get_coords(df, column, partition, indexes, offset=0.1):
    if column in categorical:
        sv = df[column][partition].sort_values()
        l, r = indexes[column][sv[sv.index[0]]], indexes[column][sv[sv.index[-1]]]+1.0
    else:
        sv = df[column][partition].sort_values()
        next_value = sv[sv.index[-1]]
        larger_values = df[df[column] > next_value][column]
        if len(larger_values) > 0:
            next_value = larger_values.min()
        l = sv[sv.index[0]]
        r = next_value
    l -= offset
    r += offset
    return l, r


'''
La funzione get_partition_rects calcola le coordinate dei rettangoli che rappresentano le partizioni di un DataFrame rispetto a due colonne specificate.
Accetta come argomenti un DataFrame df, una lista di indici di partizione partitions, i nomi di due colonne column_x e column_y,
un dizionario di indici indexes e una lista opzionale di due valori offsets che specificano gli offset da aggiungere alle coordinate (il valore predefinito è [0.1, 0.1]). 
Restituisce una lista di tuple di tuple che rappresentano le coordinate dei rettangoli delle partizioni.
'''
def get_partition_rects(df, partitions, column_x, column_y, indexes, offsets=[0.1, 0.1]):
    rects = []
    for partition in partitions:
        xl, xr = get_coords(df, column_x, partition, indexes, offset=offsets[0])
        yl, yr = get_coords(df, column_y, partition, indexes, offset=offsets[1])
        rects.append(((xl, yl),(xr, yr)))
    return rects


'''
Lla funzione get_bounds calcola i limiti di una colonna di un DataFrame. 
Accetta come argomenti un DataFrame df, il nome di una colonna column, un dizionario di indici indexes e un valore opzionale offset 
che specifica l’offset da aggiungere ai limiti (il valore predefinito è 1.0).
Restituisce una tupla di due valori che rappresentano i limiti della colonna.

Se la colonna è presente nel set categorical, i limiti vengono calcolati come 0 meno l’offset e la lunghezza del dizionario di indici corrispondente 
più l’offset. Altrimenti, i limiti vengono calcolati come il valore minimo e il valore massimo nella colonna meno e più l’offset, rispettivamente.
'''
def get_bounds(df, column, indexes, offset=1.0):
    if column in categorical:
        return 0-offset, len(indexes[column])+offset
    return df[column].min()-offset, df[column].max()+offset

indexes = build_indexes(df)
column_x, column_y = feature_columns[:2]
rects = get_partition_rects(df, finished_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])

print(rects[:10])


'''
La funzione plot_rects disegna una lista di rettangoli su un grafico. 
Accetta come argomenti un DataFrame df, un oggetto ax che rappresenta gli assi del grafico, una lista di tuple di tuple rects 
che rappresentano le coordinate dei rettangoli, i nomi di due colonne column_x e column_y e tre argomenti opzionali 
che specificano il colore del bordo (edgecolor, il valore predefinito è 'black'), il colore di riempimento 
(facecolor, il valore predefinito è 'none') e l’opacità (alpha, il valore predefinito è 0.5) dei rettangoli.
'''
def plot_rects(df, ax, rects, column_x, column_y, edgecolor='black', facecolor='none'):
    for (xl, yl),(xr, yr) in rects:
        ax.add_patch(patches.Rectangle((xl,yl),xr-xl,yr-yl,linewidth=1,edgecolor=edgecolor,facecolor=facecolor, alpha=0.5))
    ax.set_xlim(*get_bounds(df, column_x, indexes))
    ax.set_ylim(*get_bounds(df, column_y, indexes))
    ax.set_xlabel(column_x)
    ax.set_ylabel(column_y)

pl.figure(figsize=(20,20))
ax = pl.subplot(111)
plot_rects(df, ax, rects, column_x, column_y, facecolor='r')
pl.scatter(df[column_x], df[column_y])
pl.show()

'''
Le funzioni agg_categorical_column, agg_numerical_column e agg_ascii_initials vengono utilizzate per aggregare i valori di una colonna di un DataFrame.
'''
def agg_categorical_column(series):
    return [','.join(set(series))]

def agg_numerical_column(series):
    return [series.mean()]

def agg_ascii_initials(partition, nationality_col):
    ascii_values = partition[nationality_col].astype(str).tolist()
    return ','.join(ascii_values)


'''
la funzione build_anonymized_dataset costruisce un dataset anonimizzato a partire da un DataFrame e una lista di partizioni. 
Accetta come argomenti un DataFrame df, una lista di indici di partizione partitions, una lista di colonne feature_columns 
da utilizzare per l’aggregazione, il nome di una colonna sensibile sensitive_column e un intero opzionale max_partitions che specifica 
il numero massimo di partizioni da utilizzare (il valore predefinito è None). 
Restituisce un DataFrame anonimizzato.

La funzione inizia creando un dizionario vuoto aggregations. Successivamente, per ogni colonna presente nella lista feature_columns, viene selezionata la funzione di aggregazione appropriata in base al tipo di colonna (categorica o numerica) e viene aggiunta al dizionario aggregations. 
Viene quindi creata una lista vuota rows. Successivamente, per ogni partizione nella lista partitions, vengono calcolate le colonne aggregate utilizzando il metodo agg() del DataFrame e passando come argomenti il dizionario aggregations e il parametro squeeze=False.
Poi viene calcolato il conteggio dei valori sensibili nella partizione utilizzando il metodo groupby() del DataFrame e passando come argomento il nome della colonna sensibile. 
Il risultato viene aggregato utilizzando il metodo agg() e passando come argomento un dizionario che mappa il nome della colonna sensibile alla funzione 'count'.
'''
def build_anonymized_dataset(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    rows = []
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))
        if max_partitions is not None and i > max_partitions:
            break

        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column : 'count'})
        #values = pd.DataFrame(grouped_columns).iloc[0].to_dict()  ###########################

        '''
        preprocessing sui valori delle colonne age, Surname e Nationality nella partizione. 
        I valori vengono convertiti in stringhe, concatenati utilizzando la virgola come separatore e aggiunti al dizionario values con le chiavi 'age_values', 'surname_initials' e 'nationality_initials', rispettivamente. 
        Viene quindi calcolato l’intervallo dei valori in ciascuna colonna e aggiunto al dizionario values con le chiavi 'min_max_age_values', 'min_max_surname_initials' e 'min_max_nationality_initials', rispettivamente.
        '''
        age_values = df.loc[partition]['age'].astype(str).apply(lambda x: str(x)).str.cat(sep=',')
        age_values_array = df.loc[partition]['age'].apply(lambda x: int(x))
        min_max_age_values = f"{age_values_array.min()}_{age_values_array.max()}"

        surname_initials = df.loc[partition]['Surname'].astype(str).apply(lambda x: str(x)).str.cat(sep=',')
        surname_initials_array = df.loc[partition]['Surname'].apply(lambda x: int(x))
        min_max_surname_initials = f"{surname_initials_array.min()}_{surname_initials_array.max()}"

        nationality_initials = df.loc[partition]['Nationality'].astype(str).apply(lambda x: str(x)).str.cat(sep=',')
        nationality_initials_array = df.loc[partition]['Nationality'].apply(lambda x: int(x))
        min_max_nationality_initials = f"{nationality_initials_array.min()}_{nationality_initials_array.max()}"

        values = grouped_columns.to_dict()

        values.update({'age_values': age_values})
        values.update({'min_max_age_values': min_max_age_values})

        values.update({'surname_initials': surname_initials})
        values.update({'min_max_surname_initials': min_max_surname_initials})

        values.update({'nationality_initials': nationality_initials})
        values.update({'min_max_nationality_initials': min_max_nationality_initials})


        #viene calcolato il numero di record nella partizione e aggiunto al dizionario values con la chiave 'records_in_cluster'.
        values['records_in_cluster'] = len(partition)

        '''
            viene calcolato il numero di record nella partizione e aggiunto al dizionario values con la chiave 'records_in_cluster'.
            Viene quindi iterato sul dizionario values e, per ogni elemento, se il valore è una lista, viene estratto il primo elemento e sostituito al valore originale.
        '''
        for k, v in values.items():
            if isinstance(v, list):
                values[k] = v[0]

        '''
        Iterazione sul conteggio dei valori sensibili nella partizione e, per ogni valore sensibile e relativo conteggio, se il conteggio è uguale a 0, il ciclo continua con il valore successivo. 
        Altrimenti, il dizionario values viene aggiornato con il valore sensibile e il relativo conteggio e una copia del dizionario viene aggiunta alla lista rows. 
        Infine, viene creato un nuovo DataFrame a partire dalla lista rows e restituito.
        '''
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column : sensitive_value,
                'count': count,
            })
            rows.append(values.copy())
    return pd.DataFrame(rows)

dfn = build_anonymized_dataset(df, finished_partitions, feature_columns, sensitive_column)

print(dfn.sort_values(feature_columns+[sensitive_column]))
print(dfn)

'''
Le due funzioni diversity e is_l_diverse vengono utilizzate per verificare se una partizione di un DataFrame soddisfa il criterio di l-diversità rispetto a una colonna sensibile specificata.

diversity accetta come argomenti un DataFrame df, un indice di partizione partition e il nome di una colonna column. 
Restituisce il numero di valori unici nella colonna specificata nella partizione specificata.
'''

def diversity(df, partition, column):
    return len(df[column][partition].unique())

'''
La funzione is_l_diverse accetta come argomenti un DataFrame df, un indice di partizione partition, il nome di una colonna sensibile sensitive_column e un intero opzionale l che specifica il valore di l da utilizzare (il valore predefinito è 2). La funzione restituisce un valore booleano che indica se la partizione soddisfa il criterio di l-diversità. La funzione verifica se il numero di valori unici nella colonna sensibile nella partizione specificata, calcolato utilizzando la funzione diversity, è maggiore o uguale a l.
'''
def is_l_diverse(df, partition, sensitive_column, l=2):
    return diversity(df, partition, sensitive_column) >= l

finished_l_diverse_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, lambda *args: is_k_anonymous(*args) and is_l_diverse(*args))

print(len(finished_l_diverse_partitions))

column_x, column_y = feature_columns[:2]
l_diverse_rects = get_partition_rects(df, finished_l_diverse_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])

pl.figure(figsize=(20,20))
ax = pl.subplot(111)
plot_rects(df, ax, l_diverse_rects, column_x, column_y, edgecolor='b', facecolor='b')
plot_rects(df, ax, rects, column_x, column_y, facecolor='r')
pl.scatter(df[column_x], df[column_y])
pl.show()

dfl = build_anonymized_dataset(df, finished_l_diverse_partitions, feature_columns, sensitive_column)

print(dfl)
print(dfl.sort_values([column_x, column_y, sensitive_column])) #####################

# calcolo delle frequenze globali dei valori di una colonna sensibile in un DataFrame.
global_freqs = {}
total_count = float(len(df))
group_counts = df.groupby(sensitive_column)[sensitive_column].agg('count')
for value, count in group_counts.to_dict().items():
    p = count/total_count
    global_freqs[value] = p

print(global_freqs)


'''
 calcolo della t-closeness di una partizione in un dataframe. 
 La t-closeness è una misura di privacy nella anonimizzazione dei dati che garantisce che la distribuzione degli attributi sensibili in 
 qualsiasi classe di equivalenza sia vicina alla distribuzione dell’attributo nell’intero dataset. 
 La funzione prende in input un dataframe df, una partizione del dataframe, il nome di una colonna column e un dizionario global_freqs 
 contenente le frequenze globali dei valori in quella colonna. La funzione calcola la massima differenza d_max tra la frequenza locale p 
 di ogni valore nella partizione e la sua frequenza globale global_freqs[value]. Questa massima differenza viene restituita come t-closeness  della partizione.
'''
def t_closeness(df, partition, column, global_freqs):
    total_count = float(len(partition))
    d_max = None
    group_counts = df.loc[partition].groupby(column)[column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count/total_count
        d = abs(p-global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max


'''
is_t_close controlla se una partizione in un dataframe è t-chiusa rispetto a una colonna sensibile. 
La funzione prende in input un dataframe df, una partizione del dataframe, il nome di una colonna sensibile sensitive_column, un dizionario 
global_freqs contenente le frequenze globali dei valori in quella colonna e un valore soglia p. La funzione controlla prima se la colonna sensibile
è categorica e solleva un errore se non lo è. Quindi calcola la t-chiusura della partizione rispetto alla colonna sensibile utilizzando la funzione 
t_closeness definita in precedenza. Se la t-chiusura è minore o uguale al valore soglia p, 
la funzione restituisce True, indicando che la partizione è t-chiusa. Altrimenti, restituisce False.
'''
def is_t_close(df, partition, sensitive_column, global_freqs, p=0.2):
    if not sensitive_column in categorical:
        raise ValueError("this method only works for categorical values")
    return t_closeness(df, partition, sensitive_column, global_freqs) <= p

finished_t_close_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, lambda *args: is_k_anonymous(*args) and is_t_close(*args, global_freqs))

print(len(finished_t_close_partitions))

dft = build_anonymized_dataset(df, finished_t_close_partitions, feature_columns, sensitive_column)

print(dft)
df_final = dft.sort_values([column_x, column_y, sensitive_column])
df_final.to_excel('knonymyzed.xlsx', index=False)
print(df_final)

column_x, column_y = feature_columns[:2]
t_close_rects = get_partition_rects(df, finished_t_close_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])

pl.figure(figsize=(20,20))
ax = pl.subplot(111)
plot_rects(df, ax, t_close_rects, column_x, column_y, edgecolor='b', facecolor='b')
pl.scatter(df[column_x], df[column_y])
pl.show()
