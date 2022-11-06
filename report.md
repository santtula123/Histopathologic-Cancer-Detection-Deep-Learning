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

Ensimmäisessä vaiheessa verrataan tutkimuksen arkkitehtuuria kahteen muuhun arkkitehtuurin laajalla datasetillä. Arkkitehtuurit eroavat alkuperäisestä "n_filters" parametrin osalta. Ensimmäisen vaiheen jälkeen parhaaksi osoittautunut "n_filter" = [32, 32, 64, 64] arvo otettiin käyttöön toiseen vaiheeseen, jossa tehtiin kattavampaa hyperparametrien optimointia pienemmällä datasetillä. Yhteensä tässä kokeiltiin 54 eri konfiguraatiota ja parhaaksi osoittautuivat "{'act': 'relu', 'b_size': 10, 'data_path': 'data_limited', 'drop_out': X, 'epochs': 20, 'lr': 0.001, 'n_filters': [32, 32, 64, 64], 'num_classes': 1, 'optimizer': 'Adam', 'name': '1D-CNN-31'}", missä drop_out X vaihteli arvojen 0.3, 0.4 & 0.5 välillä. Kolmannessa vaiheessa parhaat kolme mallia ajettiin uudestaan laajalla datasetillä. Käytännössä malleja ajettiin kuitenkin vain kaksi, sillä yksi parhaista oli jo ajettu vaiheessa yksi. 

| Hyperparametri | Esimerkkiarvo | Selitys 
| :-: | :-: | :-: |
| data_path | data | Arvo joka määrää sen käytetäänkö laajaa vai rajoitettu testidatasettiä.
| lr | 0.001 | Oppimisnopeus.
| optimizer | Adam | Optimoija.
| epochs | 20 | Epochien lukumäärä.
| b_size | 10 | Batch koko ajoissa.
| num_classes | 1 | Ennustettavien luokkien määrä. Ulostulokerroksen neuronien määrä.
| n_filters | 32, 32, 64, 64 | Määrittelee arkkitehtuurin konvoluutio ja max-pooling kerrosten muodon.
| act | relu | Aktivointifunktio.
| drop_out | 0.3 | Verkon lopussa olevan drop_out kerroksen arvo.

## Virheanalyysi


Parhaan mallin suorituskyky evaluointidataa vastaan:

| TP | TN | FN | FP | ACC | PREC | REC | FSCORE
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 12490 | 7914 | 671 | 942 | 0.92674 | 0.92987 | 0.94902 | 0.93934 |


## Tulokset, virheanalyysi ja jatkotoimenpiteet

Parhaan mallin suorituskyky testidataa vastaan:

| TP | TN | FN | FP | ACC | PREC | REC | FSCORE
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 12185 | 7934 | 660 | 938 | 0.92642 | 0.92852 | 0.94862 | 0.93846 |

Parhaan mallin suorituskyky pysyi lähes samana testidataa vastaan. Alla oleva histogrammi havainnollistaa yllä olevia TP, TN, FN & FP arvoja vielä paremmin.

![Sekaannusmatriisi.](./resources/confusion_matrix_unseen.png)




Mallin tarkkuutta olisi mahdollista parantaa entisestään, mikäli datan laatuun kiinnitettäisiin enemmän huomiota. Erilaiset data-augmentaation metodit voisivat myös olla potentiaalisia kehityskohteita olemassa olevalle mallille. Siirto-oppiminen voisi myös olla toimiva tapa parantaa mallin suorituskykyä ja se ei vaatisi niin paljon resursseja laitteistolta, kuin nykyinen malli.