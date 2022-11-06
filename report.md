## Histopatologinen syövän tunnistus 

**Tekijä:** Santeri Moilanen, santerimoilanen@gmail.com

**Tiivistelmä**: Tutkimusartikkelin [Deep Convolutional Neural Network with TensorFlow and Keras to Classify Skin Cancer Images](https://www.researchgate.net/publication/343409875_Deep_Convolutional_Neural_Network_with_TensorFlow_and_Keras_to_Classify_Skin_Cancer_Images) neuroverkkoarkkitehtuurin replikointi ja optimointi.

## Testattu seuraavilla kirjastoilla

- Python 3.9.12
- [Tensorflow 2.7.0](https://www.tensorflow.org/overview/?hl=fi)
- [Pandas 1.4.3](https://pandas.pydata.org/)
- [NumPy 1.23.2](https://numpy.org/)
- [sklearn 1.1.1](https://scikit-learn.org/)
- [matplotlib 3.5.2](https://matplotlib.org/)
- [seaborn 0.12.1](https://seaborn.pydata.org)
- [opendatasets 0.1.22] (https://github.com/JovianML/opendatasets)


## Johdanto
Ei-melaanoma peräiset ihosyövät ovat erityisen yleisiä vaaleaihoisella eurooppalaista syn-
typerää olevilla ihmisillä. [Ferlay ym. (2021, sivu 787.)](https://doi.org/10.1002/ijc.33588) Ne ovat Pohjois-Euroopan toiseksi yleisempiä syöpiä. [Ferlay ym. (2021, sivu 785.)](https://doi.org/10.1002/ijc.33588) Altistuminen ultraviolettisäteilylle, joko
suoraan auringon tai solariumin kautta on suurin syy ihosyöpä tapauksien kasvulle [Gordon ja Rowell (2015, sivu 141)](https://doi.org/10.1097/CEJ.0000000000000056). Gordon ja Rowell [(2015, sivu 146)]( https://doi.org/10.1097/CEJ.0000000000000056) mukaan ennaltaehkäisyllä saavutettaisiin kustannussäästöjä ja saataisiin ihosyöpä vähenemään.

Voisiko ennaltaehkäisyn keinona toimia syväoppimismalli, joka osaa tunnistaa syöpää histopatologisistakuvista? Hyvin kehitetty malli voisi toimia lääkärin tukena diagnostisoidessa  syöpää potilaalta ja jopa tehdä diagnosointia lääkäriä paremmin. Tämä voisi tehostaa lääkärin työtä ja auttaa syövän löytämisessä varhaisessa vaiheessa. Tämä näkyisi terveydenhuollon kulujen vähenemisenä sekä lisääntyneinä elinvuosina ihosyöpää sairastavien potilaiden joukossa. 

## Tehtävänanto

Tarkoituksena on kehittää syväoppimismalli, joka tunnistaa histopatologisistakuvista ihosyöpää. Syväoppimismallin pohjana käytetään "Deep Convolutional Neural Network with TensorFlow and Keras to Classify Skin Cancer Images" tutkimusartikkelin arkkitehtuuria. Tätä arkkitehtuuria on tarkoitus iteroida paremmaksi arkkitehtuuri ja hyperparametrien muunnoksella.

## Data

Työssä on käytössä datasetti [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection), joka on peräisin Kagglesta julkaistusta kilpailusta. Kuvat on jaettu csv tiedoston "id" ja "label"- kenttien perusteella syöpää sisältäviksi tai ei syöpää sisältäviksi. Kuvat ovat kooltaan 96x96 pikseliä ja syöpää sisältävissä kuvissa on keskellä 32x32 pikselin kokoinen alue, joka sisältää vähintään yhden pikselin verran kasvainkudosta. Datasetti ei sisällä kaksoiskappaleita kuvista. Kuvia, joiden luokka on tiedossa on yhteensä 220 025 kappaletta. Niistä 40.5 % ovat syöpää sisältäviä kuvia ja loput 59.5 % ei syöpää sisältäviä. Loput 57 458 kuvaa ovat tarkoitettu Kaggleen kilpailua varten, jolloin niiden luokat eivät ole tiedossa.

## Esikäsittely

Datasetin kuvat jaetaan 80:10:10 suhteella opetus-, testi- ja validointidata kansioihin. Kaikkien kansioiden sisällä on kansiot "cancerous" ja "non-cancerous" sen mukaan onko kuvassa syöpää vai ei. 
Opetusdataan sovellettiin datan augmentaatiota, jolla saadaan todenmukaisia muutoksia opetusdataan. Nämä muunnokset sisälsivät sattumanvaraisen kuvan käännön vaakasuoraan, sekä kuvan siirroksia vaaka- ja pystyakseleilla. Jos esimerkiksi ajatellaan kuvaa, jossa on syöpä niin, sillä miten päin kuva on otettu ei ole väliä tai sillä onko havainto aivan kuvan keskellä.

## Metodologia

Ensimmäisessä vaiheessa verrataan tutkimuksen arkkitehtuuria kahteen muuhun arkkitehtuurin...

