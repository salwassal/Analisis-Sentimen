import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Download stopwords NLTK
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

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


# ========= TOKENISASI =========
# Fungsi untuk tokenisasi
def tokenisasi(text):
    tokens = word_tokenize(text)
    return tokens


# ========= STOPWORD REMOVEAL =========
# Inisialisasi stopword remover
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Inisialisasi stopwords dari NLTK
stop_words_nltk = set(stopwords.words('indonesian'))

# Tambahkan kata-kata khusus ke dalam daftar stopwords
more_stopwords = {'nya'}
stop_words_nltk.update(more_stopwords)

# Fungsi untuk menghapus stopwords
def remove_stopwords(tokens):
    # Hapus stopwords menggunakan Sastrawi
    filtered_tokens_sastrawi = stopword_remover.remove(' '.join(tokens)).split()
    # Gabungkan dengan stopwords dari NLTK jika diperlukan
    filtered_tokens = [token for token in filtered_tokens_sastrawi if token.lower() not in stop_words_nltk]
    return ' '.join(filtered_tokens)


# ========= STEMMING =========
# Inisialisasi stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Fungsi untuk melakukan stemming
def stemming(text):
    stemmed_text = stemmer.stem(text)
    return stemmed_text


# ========= LABELING TEXT =========
# Fungsi untuk menghitung skor sentimen
def hitung_sentimen(text, kamus_sentimen):
    skor = 0
    for kata in text.split():
        skor += kamus_sentimen.get(kata, 0)
    return skor

# Fungsi untuk mengubah skor sentimen menjadi label
def label_sentimen(skor):
    if skor > 0:
        return "positif"
    elif skor < 0:
        return "negatif"
    else:
        return "netral"

# Streamlit setup
st.set_page_config(page_title="Analisis Sentimen")

# Menu navigasi
menu = ["Home", "Prediksi Sentimen"]
choice = st.sidebar.selectbox("Pilih Menu", menu)

if choice == "Home":
    st.title("Analisis Sentimen")
    st.write("Aplikasi ini digunakan untuk melakukan analisis sentimen pada teks menggunakan metode berbasis kamus dan Bernoulli Naive Bayes.")

elif choice == "Prediksi Sentimen":
    st.title("Aplikasi Analisis Sentimen")

    # Input teks untuk analisis
    input_text = st.text_area("Masukkan teks yang ingin dianalisis", "")
    if st.button("Analisis Sentimen"):
        if input_text:
            # Preprocessing teks inputan
            cleaned_text = cleaning_data(input_text)
            normalized_text = text_normalize(cleaned_text, kamus_slang)
            tokens = tokenisasi(normalized_text)
            stopwords_removed_text = remove_stopwords(tokens)
            stemmed_text = stemming(stopwords_removed_text)

            # Hitung skor sentimen dan label sentimen
            skor_sentimen = hitung_sentimen(stemmed_text, kamus_sentimen_dict)
            label = label_sentimen(skor_sentimen)

            st.subheader("Hasil Preprocessing")
            st.write("Teks Asli:")
            st.write(input_text)
            st.write("Setelah Cleaning:")
            st.write(cleaned_text)
            st.write("Setelah Normalisasi:")
            st.write(normalized_text)
            st.write("Setelah Tokenisasi:")
            st.write(tokens)
            st.write("Setelah Stopword Removal:")
            st.write(stopwords_removed_text)
            st.write("Setelah Stemming:")
            st.write(stemmed_text)

            st.subheader("Hasil Analisis Sentimen")
            st.write(f"Skor Sentimen: {skor_sentimen}")
            st.write(f"Label Sentimen: {label}")

    # Contoh data latih
    data = pd.read_csv('supervised_comment.csv')

    df = pd.DataFrame(data)

    # Preprocessing data latih
    df['clean_text'] = df['clean_text'].apply(lambda x: stemming(remove_stopwords(tokenisasi(text_normalize(cleaning_data(x), kamus_slang)))))

    # Vectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    # Pembagian data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Inisialisasi model Bernoulli Naive Bayes
    model = BernoulliNB()

    # Melatih model
    model.fit(X_train, y_train)

    # Prediksi sentimen pada teks inputan
    if input_text:
        input_vectorized = vectorizer.transform([stemmed_text])
        prediksi_nb = model.predict(input_vectorized)[0]
        st.subheader("Hasil Prediksi dengan Bernoulli Naive Bayes")
        st.write(f"Label Sentimen: {prediksi_nb}")
