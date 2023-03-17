from pandas import read_csv, DataFrame
from re import sub, escape
from string import punctuation
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
import streamlit as st
import nltk


def clean_text(text):
    text = text.lower()
    text = sub('\[.*?\]', '', text)
    text = sub('https?://\S+|www\.\S+', '', text)
    text = sub('<.*?>+', '', text)
    text = sub('[%s]' % escape(punctuation), '', text)
    text = sub('\n', '', text)
    text = sub('\w*\d\w*', '', text)
    return text

def text_preprocessing(text):
    tokenizer = RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text

def submission(model, test_vector):
    submission_target = model.predict(test_vector)
    return submission_target[0]

@st.cache_resource
def training():
    read_and_cache_csv = st.cache_data(read_csv)
    train = read_and_cache_csv('train1.csv')


    train['location'].replace({'United States':'USA','New York':'USA',"London":'UK',"Los Angeles, CA":'USA',"Washington, D.C.":'USA',"California":'USA',"Chicago, IL":'USA',"Chicago":'USA',"New York, NY":'USA',"California, USA":'USA',"FLorida":'USA',"Nigeria":'Africa',"Kenya":'Africa',"Everywhere":'Worldwide',"San Francisco":'USA',"Florida":'USA',"United Kingdom":'UK',"Los Angeles":'USA',"Toronto":'Canada',"San Francisco, CA":'USA',"NYC":'USA',"Seattle":'USA',"Earth":'Worldwide',"Ireland":'UK',"London, England":'UK',"New York City":'USA',"Texas":'USA',"London, UK":'UK',"Atlanta, GA":'USA',"Mumbai":"India"},inplace=True)

    train['text'] = train['text'].apply(lambda x: text_preprocessing(x))


    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(train['text'])
    clf_NB_TFIDF = MultinomialNB()
    scores = model_selection.cross_val_score(clf_NB_TFIDF, train_tfidf, train["target"], cv=5, scoring="f1")
    clf_NB_TFIDF.fit(train_tfidf, train["target"])

    return [tfidf, clf_NB_TFIDF]

nltk.download('stopwords')

tfidf, clf_NB_TFIDF = training()

st.title('Определение сообщения о происшествии с помощью ИИ')
st.subheader('Данное веб-приложение является практической частью проекта команды г. Озерск шк. 32 на тему: “Искусственный интеллект в атомной отрасли”, разработанное для участия в конкурсе учебно-исследовательских работ Курчатовские Чтения 2023')
st.caption('Зачастую, сообщения от местных жителей в социальных сетях появляются раньше, чем официальные запросы в спасательные центры.')
st.caption('Если мы сможем определять, в каких из сообщений говорится о событии-катастрофе, исключая нерелевантные, например, сообщения о фильмах, мы сможем ускорить уведомление экстренных служб о чрезвычайных ситуациях и ускорить оказание необходимой помощи.')
st.caption('Подобные задачи эффективно решаются с помощью ИИ с применением машинного обучения. В нашем конкретном случае, реализация выполнена на основе методов обработки естественного языка, или NLP (Natural Language Processing) — одного из направлений искусственного интеллекта, которое работает с анализом, пониманием и генерацией живых языков.')
st.caption('Ядром приложения является обученная модель, которая определяет релевантность введенного сообщения.')
st.caption('Ниже, в поле ввода вы можете ввести сообщение и приложение сообщит является ли оно сообщением о ЧС или нет. Сообщение необходимо вводить на английском языке.')


test_id = [0]
test_keyword = [None]
test_location = [None]
test_text = [st.text_input('Введите сообщение')]

if (st.button('Определить')):
    test = DataFrame(data={'id':test_id, 'keyword':test_keyword, 'location':test_location, 'text':test_text})
    test['text'] = test['text'].apply(lambda x: text_preprocessing(x))

    test_tfidf = tfidf.transform(test["text"])

    if submission(clf_NB_TFIDF, test_tfidf) == 1:
        st.subheader(':red[Сообщение релевантно (сообщение о происшествии)]')
    else:
        st.subheader(':green[Сообщение не релевантно (сообщение не о происшествии)]')

st.caption('Сайт проекта, где находится текст с защиты: https://kurch23oz.github.io/kurch23/')
