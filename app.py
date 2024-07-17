# import streamlit as st
# import pandas as pd
# import re
# from nltk.tokenize import word_tokenize
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.metrics import confusion_matrix, classification_report

# # Fungsi untuk membersihkan data dan case folding
# def cleaning_data(text):
#     text = re.sub(r'<.*?>', ' ', text)
#     text = re.sub(r'&[a-zA-Z]+;', ' ', text)
#     text = re.sub(r'@\w+', '', text)
#     text = re.sub(r'\d+', '', text)
#     text = re.sub(r'[^\w\s]', ' ', text)
#     text = re.sub(r'_', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip()
#     text = text.lower()
#     return text

# # Fungsi untuk menghitung skor sentimen
# def hitung_sentimen(text, kamus_sentimen):
#     skor = 0
#     for kata in text.split():
#         skor += kamus_sentimen.get(kata, 0)
#     return skor

# # Fungsi untuk mengubah skor sentimen menjadi label
# def label_sentimen(skor):
#     if skor > 0:
#         return "positif"
#     elif skor < 0:
#         return "negatif"
#     else:
#         return "netral"

# # Fungsi untuk menampilkan Word Cloud
# def show_wordcloud(text, title):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     plt.figure(figsize=(10, 6))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title(title)
#     st.pyplot(plt)

# # Streamlit setup
# st.set_page_config(page_title="Analisis Sentimen", layout="wide")

# # Menu navigasi
# menu = ["Home", "Prediksi Sentimen", "Analisis Kata Baru"]
# choice = st.sidebar.selectbox("Pilih Menu", menu)

# if choice == "Home":
#     st.title("Analisis Sentimen")
#     st.write("Aplikasi ini digunakan untuk melakukan analisis sentimen pada komentar YouTube.")

# elif choice == "Prediksi Sentimen":
#     st.title("Prediksi Sentimen Komentar YouTube")

#     # Upload file komentar
#     uploaded_file = st.file_uploader("Unggah file CSV komentar", type="csv")
#     if uploaded_file is not None:
#         df_comment = pd.read_csv(uploaded_file)
#         st.write("Data Komentar:")
#         st.dataframe(df_comment.head())

#         # Proses preprocessing
#         df_comment['clean_text'] = df_comment['textDisplay'].apply(cleaning_data)

#         # Upload file kamus sentimen
#         uploaded_kamus_file = st.file_uploader("Unggah file CSV kamus sentimen", type="csv")
#         if uploaded_kamus_file is not None:
#             kamus_sentimen = pd.read_csv(uploaded_kamus_file)

#             # Konversi kamus sentimen ke dalam dictionary
#             kamus_sentimen_dict = dict(zip(kamus_sentimen['word'], kamus_sentimen['value']))

#             # Hitung skor sentimen dan label sentimen
#             df_comment['skor_sentimen'] = df_comment['clean_text'].apply(lambda x: hitung_sentimen(x, kamus_sentimen_dict))
#             df_comment['label_sentimen'] = df_comment['skor_sentimen'].apply(label_sentimen)

#             # Simpan Data Preprocessing ke file CSV
#             df_comment.to_csv('preprocessed_comment.csv', index=False, encoding='utf-8')

#             # Menghitung jumlah masing-masing label sentimen
#             sentimen_counts = df_comment['label_sentimen'].value_counts()

#             # Membuat diagram batang untuk distribusi sentimen
#             plt.figure(figsize=(10, 6))
#             ax = sns.barplot(x=sentimen_counts.index, y=sentimen_counts.values, palette='viridis')
#             plt.xlabel('Sentimen')
#             plt.ylabel('Jumlah')
#             plt.title('Distribusi Sentimen')
#             st.pyplot(plt)

#             # Membuat Word Cloud untuk masing-masing label sentimen
#             teks_positif = ' '.join(df_comment[df_comment['label_sentimen'] == 'positif']['clean_text'])
#             teks_negatif = ' '.join(df_comment[df_comment['label_sentimen'] == 'negatif']['clean_text'])
#             teks_netral = ' '.join(df_comment[df_comment['label_sentimen'] == 'netral']['clean_text'])

#             st.subheader("Word Cloud Sentimen Positif")
#             show_wordcloud(teks_positif, "Word Cloud Sentimen Positif")

#             st.subheader("Word Cloud Sentimen Negatif")
#             show_wordcloud(teks_negatif, "Word Cloud Sentimen Negatif")

#             st.subheader("Word Cloud Sentimen Netral")
#             show_wordcloud(teks_netral, "Word Cloud Sentimen Netral")

#             # Menggabungkan semua teks untuk membuat word cloud keseluruhan
#             all_text = ' '.join(df_comment['clean_text'].tolist())

#             st.subheader("Word Cloud dari Komentar")
#             wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
#             plt.figure(figsize=(10, 6))
#             plt.imshow(wordcloud, interpolation='bilinear')
#             plt.axis('off')
#             plt.title('Word Cloud dari Komentar')
#             st.pyplot(plt)

#             # Pembagian data menjadi data latih dan data uji
#             X = df_comment['clean_text']
#             y = df_comment['label_sentimen']

#             # Vectorizer
#             vectorizer = CountVectorizer()
#             X_vectorized = vectorizer.fit_transform(X)

#             # Pembagian data latih dan uji
#             X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

#             # Inisialisasi model Bernoulli Naive Bayes
#             model = BernoulliNB()

#             # Melatih model
#             model.fit(X_train, y_train)

#             # Prediksi sentimen pada data uji
#             y_pred = model.predict(X_test)

#             st.subheader("Hasil Prediksi")
#             st.write("Confusion Matrix:")
#             cm = confusion_matrix(y_test, y_pred)
#             st.write(cm)

#             st.write("Classification Report:")
#             report = classification_report(y_test, y_pred, output_dict=True)
#             st.write(pd.DataFrame(report).transpose())

# elif choice == "Analisis Kata Baru":
#     st.title("Analisis Kata Baru")

#     input_text = st.text_area("Masukkan teks yang ingin dianalisis", "")
#     if st.button("Analisis"):
#         if input_text:
#             # Preprocessing teks inputan
#             cleaned_text = cleaning_data(input_text)
            
#             # Stemming menggunakan Sastrawi
#             factory = StemmerFactory()
#             stemmer = factory.create_stemmer()
#             stemmed_text = stemmer.stem(cleaned_text)

#             # Hitung skor sentimen dan label sentimen
#             skor_sentimen = hitung_sentimen(stemmed_text, kamus_sentimen_dict)
#             label = label_sentimen(skor_sentimen)

#             st.subheader("Hasil Preprocessing")
#             st.write("Teks Asli:")
#             st.write(input_text)
#             st.write("Teks Setelah Preprocessing:")
#             st.write(stemmed_text)

#             st.subheader("Hasil Analisis Sentimen")
#             st.write(f"Skor Sentimen: {skor_sentimen}")
#             st.write(f"Label Sentimen: {label}")


# ======================================================================

# import streamlit as st
# import pandas as pd
# import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.metrics import confusion_matrix, classification_report

# # Fungsi untuk membersihkan data dan case folding
# def cleaning_data(text):
#     text = re.sub(r'<.*?>', ' ', text)
#     text = re.sub(r'&[a-zA-Z]+;', ' ', text)
#     text = re.sub(r'@\w+', '', text)
#     text = re.sub(r'\d+', '', text)
#     text = re.sub(r'[^\w\s]', ' ', text)
#     text = re.sub(r'_', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip()
#     text = text.lower()
#     return text

# # Fungsi untuk menghitung skor sentimen
# def hitung_sentimen(text, kamus_sentimen):
#     skor = 0
#     for kata in text.split():
#         skor += kamus_sentimen.get(kata, 0)
#     return skor

# # Fungsi untuk mengubah skor sentimen menjadi label
# def label_sentimen(skor):
#     if skor > 0:
#         return "positif"
#     elif skor < 0:
#         return "negatif"
#     else:
#         return "netral"

# # Fungsi untuk menampilkan Word Cloud
# def show_wordcloud(text, title):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     plt.figure(figsize=(10, 6))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title(title)
#     st.pyplot(plt)

# # Streamlit setup
# st.set_page_config(page_title="Analisis Sentimen", layout="wide")

# # Menu navigasi
# menu = ["Home", "Prediksi Sentimen"]
# choice = st.sidebar.selectbox("Pilih Menu", menu)

# if choice == "Home":
#     st.title("Analisis Sentimen")
#     st.write("Aplikasi ini digunakan untuk melakukan analisis sentimen pada komentar YouTube.")

# elif choice == "Prediksi Sentimen":
#     st.title("Prediksi Sentimen Komentar YouTube")

#     # Upload file komentar
#     uploaded_file = st.file_uploader("Unggah file CSV komentar", type="csv")
#     if uploaded_file is not None:
#         df_comment = pd.read_csv(uploaded_file)
#         st.write("Data Komentar:")
#         st.dataframe(df_comment.head())

#         # Proses preprocessing
#         df_comment['clean_text'] = df_comment['textDisplay'].apply(cleaning_data)

#         # Upload file kamus sentimen
#         uploaded_kamus_file = st.file_uploader("Unggah file CSV kamus sentimen", type="csv")
#         if uploaded_kamus_file is not None:
#             kamus_sentimen = pd.read_csv(uploaded_kamus_file)

#             # Konversi kamus sentimen ke dalam dictionary
#             kamus_sentimen_dict = dict(zip(kamus_sentimen['word'], kamus_sentimen['value']))

#             # Hitung skor sentimen dan label sentimen
#             df_comment['skor_sentimen'] = df_comment['clean_text'].apply(lambda x: hitung_sentimen(x, kamus_sentimen_dict))
#             df_comment['label_sentimen'] = df_comment['skor_sentimen'].apply(label_sentimen)

#             # Simpan Data Preprocessing ke file CSV
#             df_comment.to_csv('preprocessed_comment.csv', index=False, encoding='utf-8')

#             # Menghitung jumlah masing-masing label sentimen
#             sentimen_counts = df_comment['label_sentimen'].value_counts()

#             # Membuat diagram batang untuk distribusi sentimen
#             plt.figure(figsize=(10, 6))
#             ax = sns.barplot(x=sentimen_counts.index, y=sentimen_counts.values, palette='viridis')
#             plt.xlabel('Sentimen')
#             plt.ylabel('Jumlah')
#             plt.title('Distribusi Sentimen')
#             st.pyplot(plt)

#             # Membuat Word Cloud untuk masing-masing label sentimen
#             teks_positif = ' '.join(df_comment[df_comment['label_sentimen'] == 'positif']['clean_text'])
#             teks_negatif = ' '.join(df_comment[df_comment['label_sentimen'] == 'negatif']['clean_text'])
#             teks_netral = ' '.join(df_comment[df_comment['label_sentimen'] == 'netral']['clean_text'])

#             st.subheader("Word Cloud Sentimen Positif")
#             show_wordcloud(teks_positif, "Word Cloud Sentimen Positif")

#             st.subheader("Word Cloud Sentimen Negatif")
#             show_wordcloud(teks_negatif, "Word Cloud Sentimen Negatif")

#             st.subheader("Word Cloud Sentimen Netral")
#             show_wordcloud(teks_netral, "Word Cloud Sentimen Netral")

#             # Menggabungkan semua teks untuk membuat word cloud keseluruhan
#             all_text = ' '.join(df_comment['clean_text'].tolist())

#             st.subheader("Word Cloud dari Komentar")
#             wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
#             plt.figure(figsize=(10, 6))
#             plt.imshow(wordcloud, interpolation='bilinear')
#             plt.axis('off')
#             plt.title('Word Cloud dari Komentar')
#             st.pyplot(plt)

#             # Pembagian data menjadi data latih dan data uji
#             X = df_comment['clean_text']
#             y = df_comment['label_sentimen']

#             # Vectorizer
#             vectorizer = CountVectorizer()
#             X_vectorized = vectorizer.fit_transform(X)

#             # Pembagian data latih dan uji
#             X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.1, random_state=42)

#             # Inisialisasi model Bernoulli Naive Bayes
#             model = BernoulliNB()

#             # Melatih model
#             model.fit(X_train, y_train)

#             # Prediksi sentimen pada data uji
#             y_pred = model.predict(X_test)

#             st.subheader("Hasil Prediksi")
#             st.write("Confusion Matrix:")
#             cm = confusion_matrix(y_test, y_pred)
#             st.write(cm)

#             st.write("Classification Report:")
#             report = classification_report(y_test, y_pred, output_dict=True)
#             st.write(pd.DataFrame(report).transpose())


# =========================================================

# import streamlit as st
# import pickle

# # Muat model dan vectorizer
# with open('model_sentimen.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# with open('vectorizer.pkl', 'rb') as vec_file:
#     vectorizer = pickle.load(vec_file)

# # Judul aplikasi
# st.title('Aplikasi Analisis Sentimen')

# # Input dari pengguna
# user_input = st.text_area('Masukkan teks yang ingin dianalisis:', '')

# if st.button('Analisis Sentimen'):
#     if user_input:
#         # Transformasi teks pengguna menggunakan vectorizer
#         input_vectorized = vectorizer.transform([user_input])
        
#         # Prediksi sentimen
#         prediction = model.predict(input_vectorized)
#         sentiment_dict = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
#         sentiment = sentiment_dict[prediction[0]]
        
#         # Tampilkan hasil
#         st.write(f'Sentimen: {sentiment}')
#     else:
#         st.write('Masukkan teks untuk dianalisis.')

# =============================

import streamlit as st
import pandas as pd
import re
import nltk
# import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
from collections import Counter

# Download stopwords dan punkt
nltk.download('stopwords')
nltk.download('punkt')

# Baca kamus slang dan kamus sentimen
kamus_slang = pd.read_csv('kamus_slang.csv')
kamus_sentimen = pd.read_csv('kamus_lexicon.csv')
kamus_sentimen_dict = dict(zip(kamus_sentimen['word'], kamus_sentimen['value']))

# ========= CLEANING DATA =========
# Fungsi untuk membersihkan data
def cleaning_data(text):
    text = re.sub(r'<.*?>', ' ', text)  # Menghapus tag HTML
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # Menghapus entitas HTML
    text = re.sub(r'@\w+', '', text)  # Menghapus username yang dimulai dengan '@'
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'[^\w\s]', ' ', text)  # Hapus tanda baca
    text = re.sub(r'_', '', text) # Menghapus underscore
    text = re.sub(r'\s+', ' ', text)  # Mengganti spasi berlebih dengan satu spasi
    text = text.strip()  # Menghapus spasi di awal dan akhir teks
    text = text.lower() # Case Folding
    return text


# ========= NORMALISASI KATA =========
def tambah_kata_slang(kamus_slang, kata_baru):
    # Menggabungkan kamus_slang dengan kata_baru
    kamus_slang = pd.concat([kamus_slang, kata_baru], ignore_index=True)
    return kamus_slang

# Contoh kata-kata baru
kata_baru = pd.DataFrame({
    'slang': ['mudamudahan', 'prof', 'mntap', 'menhargai', 'orgnya', 'diatopsi', 'otopsi', 'dioutopsi', 'suanida', 'menegak',
              'waktu²', 'dipertanya', 'narsumnya', 'menliat', 'nglantur', 'dipriksa', 'jngat', 'nehhhh', 'hbt', 'sihanida',
              'cepatnx', 'jessika', 'compiuter', 'degitalisasi', 'tehnologi', 'donnnkkk', 'daghhh', 'nebak', 'nnya', 'berap', 'dtnya',
              'manifulasi', 'leibh', 'apapa', 'jadiii', 'nangani', 'ytta', 'mw', 'x', 'bebasskn', 'kejangalan'],

    'formal': ['mudah mudahan', 'profesor', 'mantap', 'menghargai', 'orangnya', 'diautopsi', 'autopsi', 'diautopsi', 'sianida', 'menegakkan',
               'waktu waktu', 'dipertanyakan', 'narsumbernya', 'melihat', 'melantur', 'diperiksa', 'ingat', 'nih', 'hebat', 'sianida',
               'cepatnya', 'jessica', 'komputer', 'digitalisasi', 'teknologi', 'dong', 'deh', 'tebak', 'tanya', 'berapa', 'ditanya',
               'manipulasi', 'lebih', 'apa apa', 'jadi', 'menangani', 'yang tahu tahu aja', 'mau', 'kali', 'bebaskan', 'kejanggalan']
})

kamus_slang = tambah_kata_slang(kamus_slang, kata_baru)

# Fungsi untuk normalisasi kata slang
def text_normalize(text, kamus_slang):
    slang_dict = dict(zip(kamus_slang['slang'], kamus_slang['formal']))
    text = ' '.join([slang_dict.get(word, word) for word in text.split()])
    text = cleaning_data(text)
    return text

# Fungsi untuk menghapus stopwords
def remove_stopwords(tokens, stop_words_nltk):
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words_nltk]
    return ' '.join(filtered_tokens)

# Fungsi untuk stemming
def stemming(text, stemmer):
    return stemmer.stem(text)

# Fungsi untuk menghitung skor sentimen dan menampilkan bobot kata
def hitung_sentimen(text, kamus_sentimen):
    skor = 0
    bobot_kata = []
    for kata in text.split():
        if kata in kamus_sentimen:
            skor_kata = kamus_sentimen[kata]
            bobot_kata.append((kata, skor_kata))
            skor += skor_kata
        else:
            bobot_kata.append((kata, 0))
    return skor, bobot_kata

# Fungsi untuk mengubah skor sentimen menjadi label
def label_sentimen(skor):
    if skor > 0:
        return "positif"
    elif skor < 0:
        return "negatif"
    else:
        return "netral"

# Fungsi untuk menampilkan Word Cloud
def show_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

# Fungsi untuk menghitung frekuensi kata
def hitung_frekuensi(teks):
    tokens = nltk.word_tokenize(teks)
    frekuensi = Counter(tokens)
    return frekuensi

# Fungsi untuk menampilkan n kata yang paling sering muncul
def tampilkan_kata_tersering(frekuensi, n=10):
    kata_tersering = frekuensi.most_common(n)
    for kata, jumlah in kata_tersering:
        st.write(f"{kata}: {jumlah}")

# Load dan proses data
dir_comment = 'youtube_commentsnew.csv'
df_comment = pd.read_csv(dir_comment)

df_comment['cleaning_data'] = df_comment['textDisplay'].apply(cleaning_data)
df_comment['normalized_text'] = df_comment['cleaning_data'].apply(text_normalize, slang_dict=slang_dict)
df_comment['tokens'] = df_comment['normalized_text'].apply(nltk.word_tokenize)

# Inisialisasi stopwords
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()
stop_words_nltk = set(stopwords.words('indonesian'))
more_stopwords = {'nya'}
stop_words_nltk.update(more_stopwords)

df_comment['stopwords'] = df_comment['tokens'].apply(remove_stopwords, stop_words_nltk=stop_words_nltk)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
df_comment['clean_text'] = df_comment['stopwords'].apply(stemming, stemmer=stemmer)

df_comment.to_csv('comment_clean.csv', index=False, encoding='utf-8')

df_comment_clean = pd.read_csv('comment_clean.csv')
df_comment_clean['skor_sentimen'], df_comment_clean['bobot_kata'] = zip(*df_comment_clean['clean_text'].apply(lambda x: hitung_sentimen(x, kamus_sentimen_dict)))
df_comment_clean['label_sentimen'] = df_comment_clean['skor_sentimen'].apply(label_sentimen)
df_comment_clean.to_csv('supervised_comment.csv', index=False, encoding='utf-8')

sentimen_counts = df_comment_clean['label_sentimen'].value_counts()

all_text = df_comment_clean['clean_text'].tolist()
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(all_text)

X_train, X_test, y_train, y_test = train_test_split(X, df_comment_clean['label_sentimen'], test_size=0.1, random_state=42)
bnb_model = BernoulliNB(alpha=1.0)
bnb_model.fit(X_train, y_train)
y_pred_bnb = bnb_model.predict(X_test)

cm_bnb = confusion_matrix(y_test, y_pred_bnb, labels=['positif', 'negatif', 'netral'])

# Streamlit interface
st.title("Aplikasi Analisis Sentimen Komentar YouTube")

menu = st.sidebar.selectbox("Menu", options=["Home", "Prediksi Sentimen"])

if menu == "Home":
    st.write("### Deskripsi Aplikasi")
    st.write("""
    Aplikasi ini digunakan untuk menganalisis sentimen komentar pada video YouTube. 
    Aplikasi ini menggunakan pendekatan Naive Bayes dan analisis berbasis leksikon untuk menentukan sentimen dari setiap komentar.
    Anda dapat memasukkan komentar pada menu 'Prediksi Sentimen' untuk mengetahui sentimennya.
    """)

    st.write("### Distribusi Sentimen")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=sentimen_counts.index, y=sentimen_counts.values, palette='viridis')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    plt.title('Distribusi Sentimen')

    for i in ax.containers:
        ax.bar_label(i)

    st.pyplot(plt)

    st.write("### Confusion Matrix Bernoulli Naive Bayes")
    sns.heatmap(cm_bnb, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix Bernoulli Naive Bayes')
    st.pyplot(plt)

    st.write("### Word Cloud dari Seluruh Data Komentar")
    all_text_combined = ' '.join(all_text)
    show_wordcloud(all_text_combined, "Word Cloud dari Komentar")

    frekuensi_all = hitung_frekuensi(all_text_combined)
    st.write("### Kata yang Sering Muncul dalam Seluruh Data:")
    tampilkan_kata_tersering(frekuensi_all)

elif menu == "Prediksi Sentimen":
    st.write("### Prediksi Sentimen Komentar")
    input_teks = st.text_area("Masukkan teks komentar:")

    if st.button("Prediksi"):
        teks_clean = cleaning_data(input_teks)
        teks_normalized = text_normalize(teks_clean, slang_dict)
        teks_tokens = nltk.word_tokenize(teks_normalized)
        teks_stopwords = remove_stopwords(teks_tokens, stop_words_nltk)
        teks_stemmed = stemming(teks_stopwords, stemmer)
        teks_vectorized = vectorizer.transform([teks_stemmed])
        prediksi = bnb_model.predict(teks_vectorized)
        st.write("Sentimen dari teks yang dimasukkan adalah:", prediksi[0])