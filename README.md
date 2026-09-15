# Forløb: Klassifikation med ANN og CNN på MNIST-datasættet

I det følgende introducerer vi **Artificial Neural Networks (ANN)** og
**Convolutional Neural Networks (CNN)** ved hjælp af
**MNIST-datasættet**, som indeholder billeder af håndskrevne cifre fra 0
til 9.

Forløbet viser først, hvordan et ANN kan klassificere cifrene, og
derefter hvordan et CNN kan udnytte billedernes todimensionelle
struktur.

## Formål


1.  forklare grundideen i et neuralt netværk.
2.  bygge og træne et simpelt ANN.
3.  forklare forskellen på et ANN og et CNN.
4.  bygge og træne et simpelt CNN.
5.  forklare begreber som **vægt**, **bias**, **aktiveringsfunktion**,
    **epoke**, **accuracy** og **loss**.
6.  evaluere og sammenligne ANN- og CNN-modeller.

------------------------------------------------------------------------

## MNIST-datasættet

MNIST er et datasæt med billeder af håndskrevne cifre.

-   Der er **60.000 træningsbilleder**.
-   Der er **10.000 testbilleder**.
-   Hvert billede er **28 × 28 pixels**.
-   Hvert billede tilhører én af **10 klasser**: 0, 1, 2, ..., 9.
-   Hver pixel har oprindeligt en værdi mellem **0 og 255**.

MNIST findes indbygget i TensorFlow/Keras.

------------------------------------------------------------------------

## 1. Indlæs og undersøg MNIST-data

``` python
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Indlæs MNIST-datasættet
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Træningsdata:", X_train.shape)
print("Testdata:", X_test.shape)
```

`X_train` indeholder billederne, som modellen skal trænes på, mens
`y_train` indeholder det korrekte ciffer for hvert billede.

Tilsvarende indeholder `X_test` og `y_test` de data, som senere kan
bruges til at undersøge, hvor godt modellen klarer nye billeder.

### Se nogle af billederne

``` python
fig, axes = plt.subplots(1, 5, figsize=(10, 3))

for i, ax in enumerate(axes):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(f"Ciffer: {y_train[i]}")
    ax.axis("off")

plt.show()
```

### Opgave

Prøv at ændre, hvilke billeder der vises. Kan du selv genkende alle
cifrene?

------------------------------------------------------------------------

## 2. Normaliser data

Pixelværdierne ligger oprindeligt mellem 0 og 255. Vi ændrer dem, så de
ligger mellem 0 og 1.

``` python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

Det kaldes **normalisering**.

------------------------------------------------------------------------

# Del 1: Artificial Neural Network (ANN)

## 3. Hvad er et ANN?

Et ANN består af neuroner organiseret i lag.

I vores model bruger vi:

1.  et input,
2.  et skjult lag,
3.  et outputlag.

Et billede består af 28 × 28 pixels:

``` text
28 × 28 = 784 pixelværdier
```

I et fuldt forbundet lag bliver hver inputværdi sendt videre til alle
neuroner i det næste lag.

Hver forbindelse har sin egen **vægt**.

En neuron beregner grundlæggende en vægtet sum:

``` text
input × vægt + input × vægt + ... + bias
```

Derefter anvendes typisk en **aktiveringsfunktion**.

------------------------------------------------------------------------

## 4. Byg et simpelt ANN

``` python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

ann_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
```

### `Flatten`

``` python
Flatten(input_shape=(28, 28))
```

omdanner billedet fra en 28 × 28 matrix til en række med 784 værdier.

### Det skjulte lag

``` python
Dense(128, activation="relu")
```

Laget har 128 neuroner.

Hver af de 784 inputværdier er forbundet med alle 128 neuroner, og hver
forbindelse har sin egen vægt.

`relu` er lagets såkaldte aktiveringsfunktion.

### Outputlaget

``` python
Dense(10, activation="softmax")
```

Der er 10 neuroner i outputlaget -- én for hvert muligt ciffer.

`softmax` omdanner outputtet til en sandsynlighedsfordeling over de 10
klasser.

------------------------------------------------------------------------

## 5. Kompilér ANN-modellen

``` python
ann_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

Her bestemmes blandt andet:

-   hvordan vægtene skal justeres (`optimizer`)
-   hvordan modellens fejl skal måles (`loss`)
-   hvilken evalueringsmetrik vi vil følge (`accuracy`)

------------------------------------------------------------------------

## 6. Træn ANN-modellen

``` python
ann_history = ann_model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.1
)
```

En **epoke** er én gennemgang af hele træningsdatasættet.

`epochs=5` betyder derfor, at modellen gennemgår træningsdata fem gange.

`validation_split=0.1` betyder, at 10 % af træningsdataene holdes
tilbage og bruges som valideringsdata under træningen. Testdataene
holdes dermed adskilt til den afsluttende evaluering.

Under træningen kan du blandt andet se:

``` text
accuracy
loss
val_accuracy
val_loss
```

-   **accuracy**: andelen af korrekte klassifikationer på de data,
    modellen træner på.
-   **loss**: modellens tab på træningsdata.
-   **val_accuracy**: nøjagtigheden på valideringsdata.
-   **val_loss**: tabet på valideringsdata.

------------------------------------------------------------------------

## 7. Evaluér ANN-modellen

``` python
loss, accuracy = ann_model.evaluate(X_test, y_test)

print(f"ANN-modellens nøjagtighed: {accuracy:.2f}")
```

Nu bruges testdataene til at undersøge, hvor godt den færdigtrænede
model generaliserer til data, den ikke er trænet på.

------------------------------------------------------------------------

# Del 2: Convolutional Neural Network (CNN)

## 8. Hvorfor bruge et CNN?

ANN-modellen flader billedet ud til 784 værdier. Dermed udnyttes
billedernes todimensionelle struktur ikke direkte.

Et CNN arbejder derimod med billedet som et billede.

CNN'et bruger **convolutional filters** til at undersøge små områder af
billedet.

Et filter kan eksempelvis være:

``` text
3 × 3
```

Filterets værdier er **vægte**, som modellen lærer under træningen.

Det samme filter anvendes forskellige steder på billedet. Dermed kan
modellen eksempelvis lære at reagere på bestemte lokale mønstre.

------------------------------------------------------------------------

## 9. Klargør billederne til CNN

CNN'et forventer, at billederne også har en kanal-dimension.

MNIST er gråtonebilleder og har derfor én kanal.

``` python
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
```

Et billede har nu formen:

``` text
28 × 28 × 1
```

------------------------------------------------------------------------

## 10. Byg CNN-modellen

``` python
from tensorflow.keras.layers import Conv2D, MaxPooling2D

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
```

------------------------------------------------------------------------

## 11. Convolutional lag

Den første linje er:

``` python
Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1))
```

`32` betyder, at laget lærer **32 forskellige filtre**.

Hvert filter er i dette tilfælde:

``` text
3 × 3
```

Filtrene bevæger sig hen over billedet og leder efter lokale mønstre.

I begyndelsen er filtervægtningen ikke færdiglært. Under træningen
justeres filtervægtningerne, så de bliver nyttige til
klassifikationsopgaven.

------------------------------------------------------------------------

## 12. Pooling-lag

Efter convolutional-laget kommer:

``` python
MaxPooling2D((2, 2))
```

Max pooling opdeler outputtet i små 2 × 2 områder og beholder den
største værdi fra hvert område.

Det reducerer mængden af data, samtidig med at stærke aktiveringer
bevares.

------------------------------------------------------------------------

## 13. Flere convolutional lag

Det næste convolutional-lag er:

``` python
Conv2D(64, (3, 3), activation="relu")
```

Her bruges 64 filtre.

Senere convolutional-lag kan kombinere information fra tidligere lag og
dermed lære mere komplekse mønstre.

------------------------------------------------------------------------

## 14. Fra convolution til klassifikation

Efter convolution og pooling har vi:

``` python
Flatten()
Dense(128, activation="relu")
Dense(10, activation="softmax")
```

`Flatten()` omdanner de feature maps, CNN'et har produceret, til én
række værdier.

Disse sendes gennem et fuldt forbundet lag, før outputlaget producerer
sandsynligheder for de 10 cifre.

------------------------------------------------------------------------

## 15. Kompilér CNN-modellen

``` python
cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

------------------------------------------------------------------------

## 16. Træn CNN-modellen

``` python
cnn_history = cnn_model.fit(
    X_train_cnn,
    y_train,
    epochs=5,
    validation_split=0.1
)
```

Også her gennemgår modellen træningsdata fem gange.

Efter hver epoke kan vi følge udviklingen i `accuracy`, `loss`,
`val_accuracy` og `val_loss`.

------------------------------------------------------------------------

## 17. Evaluér CNN-modellen

``` python
loss, accuracy = cnn_model.evaluate(X_test_cnn, y_test)

print(f"CNN-modellens nøjagtighed: {accuracy:.2f}")
```

------------------------------------------------------------------------

# ANN og CNN sammenlignet

  -----------------------------------------------------------------------
  ANN                                 CNN
  ----------------------------------- -----------------------------------
  Billedet flades ud                  Bevarer billedets 2D-struktur

  Fuldt forbundne lag                 Convolutional- og pooling-lag

  Hver forbindelse har sin egen vægt  Filtervægte genbruges forskellige
                                      steder i billedet

  Ser ikke direkte lokale naboforhold Udnytter lokale mønstre

  Kan bruges til billedklassifikation Særligt velegnet til
                                      billedklassifikation
  -----------------------------------------------------------------------

CNN er stadig et **neuralt netværk**. Den afgørende forskel er især,
hvordan de første lag behandler inputdataene.

------------------------------------------------------------------------



------------------------------------------------------------------------

# Refleksionsopgaver

1.  Hvorfor normaliserer vi pixelværdierne?
2.  Hvad sker der med et 28 × 28 billede i `Flatten()`?
3.  Hvad betyder det, at et lag er `Dense`?
4.  Hvad er en vægt?
5.  Hvad er en epoke?
6.  Hvad er forskellen på `accuracy` og `loss`?
7.  Hvad er forskellen på `accuracy` og `val_accuracy`?
8.  Hvad er et convolutional filter?
9.  Hvorfor genbruges de samme filtervægte forskellige steder i
    billedet?
10. Hvad gør max pooling?
11. Hvorfor er CNN særligt velegnet til billeder?
12. Sammenlign ANN'ets og CNN'ets nøjagtighed. Hvilken model klarer sig
    bedst?

------------------------------------------------------------------------

# Eksperimenter

## Eksperiment 1 -- antal epoker

Ændr:

``` python
epochs=5
```

til eksempelvis:

``` python
epochs=1
```

eller:

``` python
epochs=10
```

Hvordan påvirker det modellens præstation?

## Eksperiment 2 -- antal neuroner

Ændr ANN'ets skjulte lag:

``` python
Dense(128, activation="relu")
```

til eksempelvis 32 eller 256 neuroner.

Hvad sker der?

## Eksperiment 3 -- antal filtre

Ændr CNN'ets første convolutional-lag:

``` python
Conv2D(32, (3, 3), activation="relu")
```

Prøv eksempelvis 16 eller 64 filtre.

Hvordan påvirker det træningstid og nøjagtighed?

------------------------------------------------------------------------

# Opsamling

I det ovenstående har vi bevæget os fra et almindeligt kunstigt neuralt netværk
til et convolutional neural network.

Den grundlæggende idé er den samme: Modellen lærer **vægte** ud fra
træningsdata.

I ANN'et er neuronerne i de fuldt forbundne lag forbundet med alle
værdier fra det foregående lag.

I CNN'et organiseres en del af vægtene i **filtre**, som genbruges
forskellige steder på billedet. Det gør CNN'et i stand til at udnytte
den rumlige struktur i billeder og lære lokale mønstre.
