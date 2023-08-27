---
title: Relazione Progetto Gpu Computing
subtite: Raytracing
author: Davide Cologni
header-includes:
   - \usepackage{tikz}
---

\tableofcontents

# Introduzione
  Lo scopo di questo progetto è di implementare un raytracer usando il modello di calcolo CUDA.
  Un raytracer è un programma che genera un immagine calcolando il colore del singolo pixel basandosi sul percorso fatto dalla luce.
  Ogni singolo pixel dell'immagine viene calcolato indipendentemente dagli altri, questo rende il problema ottimo per essere parallelizzato.

# Descrizione dell'algoritmo
  L'algoritmo di raytracing simula il comportamento della luce, calcolando il percorso fatto dal singolo fotone a contatto con le superfici.
  La mia implementazione dell'algoritmo legge i parametri della scena da un file di testo e esporta un'immagine in formato bitmap.
  La scena può contenere
     - dei punti luce di ogni colore e raggio,
     - geometrie primitive, quali sfere e piani
     - mesh
  Il formato del file di testo è descritto in ... formato della scena.
  
  
## Ray Tracing
   Per ogni pixel dell'immagine l'algoritmo simula il percorso del fotone tracciando un raggio nella scena.
   Ogni volta che il raggio viene tracciato nella scena si calcola quale è l'oggetto più vicino con cui collide e in che punto.
   Il colore del pixel viene calcolato sommando due componenti: componente diffusiva e la componente riflessiva.
   La componente diffusiva è determinata dal colore del materiale dell'oggetto colpito e dalle fonti di luci che raggiungono quel punto.
   La componente riflessiva viene calcolata ricorsivamente tracciando un altro raggio nella scena, essa simula il rimbalzo del fotone sull'oggetto.
   La direzione del rimbalzo dipende dalla normale dell'oggetto colpito nel punto e dal tipo di materiale.
   Ho implementato due tipi di materiale: plastico e metallico.
   Il materiale plastico rappresenta un tipo di riflessione quasi totalmente diffussivo, in questo caso la luce viene riflessa in più direzioni.

   \begin{center}
   \begin{tikzpicture}
   \centering
   \draw[black, thick] (-2,-1) -- (2,-1);
   \draw[->, black, thick] (-2, 1) -- (0,-1);
   \draw[->, black, thick] (0, -1) -- (2, 1);
   \end{tikzpicture}
   \end{center}

   Il materiale metallico invece riflette la luce in un solo angolo, specchiando la direzione della luce rispetto alla normale.
   
   \begin{center}
   \begin{tikzpicture}
   \draw[black, thick] (-2,-1) -- (2,-1);
   \draw[->, blue, very thick] (-2, 1) -- (0,-1);

   \draw[->, black, thin] (0, -1) -- (2, 1);
   \draw[->, black, thin] (0, -1) -- (0, 2);
   \draw[->, black, thin] (0, -1) -- (1, 2);
   \draw[->, black, thin] (0, -1) -- (-1, 2);
   \end{tikzpicture}
   \end{center}

   
## Formato della scena
Per descrivere gli oggetti nella scena ho usato un semplice formato testuale.
La prima linea del file è l'intetazione e specifica le dimensioni dell'immagine:   
```
   size 600 600
```

Per descrivere un punto luce    
```
   light (0, 0, 0) red  
```

dove `(0, 0, 0)` è la posizione composta da tre numeri decimali, e `red` è il colore che si può anche descrivere come vettore rgb (`(1, 0, 0)` in questo caso).    

Per aggiungere le geometrie di base:
```
   sphere (-2.5, -2.5, -10) .5 white
```

Dopo la direttiva `sphere` segue la posizione all'interno della scena, poi il suo raggio e infine il colore del materiale se non diversamente specificato il materiale è plastico, per specificare un materiale metallico si può scrivere `metal:colore`

Per descrivere i piani, basta specificare `plane` il suo centro, seguita della normale del piano e infine la descrizione del materiale 
es.   
```
plane (-7, 0, -4) (1, 0, 0) metal:white
```

Infine per caricare modelli all'interno della scena si può usare la direttiva `model` seguita dal percorso del file obj.
```
model "stanford_bunny.obj"
```


# Utilizzo del modello di programmazione CUDA
## Suddivisione dei thread
Per sua natura un algoritmo di raytracing è un problema altamente parallelizzabile: ogni pixel viene calcolato indipendentemente dagli altri. Per questo motivo ho deciso di assegnare un pixel a un thread.   
Per sfruttare al meglio il modello di programmazione CUDA ho deciso di suddividere i thread in blocchi corrispondenti a sezioni di pixel 8x8 questo perchè intuitivamente, raggi vicini tra loro e collideranno più probabilmente con lo stesso oggetto e anche la direzione del loro rimbalzo sarà più simile rispetto ai raggi più distanti.   
Se due raggi collidono con gli stessi oggetti a ogni rimbalzo i loro threads avranno lo stesso percorso di esecuzione, quindi minimizzare il numero di variazioni nelle collisioni nei thread dello stesso blocco significa minimizzare la divergenza.
Questa suddivisione non sempre riesce a beneficiare di questo principio: se si collide con un oggetto con materiale diffusivo, la direzione di rimbalzo viene campionata secondo una distribuzione uniforme.   

## Allocazione della scena
La scena è stata allocata usando la unified memory.
**---- TODO: completare questa sezione**     
L'oggetto scena contiene .... descrizione oggetto scena
Per migliorare le prestazioni .... tutti allocati insieme
Suddivisione della memoria .... 


## Uso di cu_rand per materiale plastico
Ogni kernel prende come parametro un puntatore (costante, sola lettura) all'oggetto scena, un buffer di memoria contenente tutti i pixel dell'immagine con le sue dimensioni (larghezza e altezza) e un oggetto `curandState`

## Classe Vec3
Per implementare tutte le operazioni vettoriali come somma, prodotto, prodotto scalare ecc... ho implementato una classe Vec3 con 3 decimali x, y e z.    
L'implementazione della classe è abbastanza ovvia l'unica nota è che tutti i metodi sono stati decorati con le direttive `__host__` e `__device__`.    
In questo modo questa classe è usabile sia da codice host, sia dalla GPU.    

## Streams (?)

## Pinned Memory
Per migliorare il tempo di trasferimento dei dati dalla GPU alla CPU con l'uso degli stream, e prevenire la sincronizzazione implicita, il buffer di memoria che contiene i valori dell'immagine è stato allocato nella pinned memory.
La pinned memory è una porzione della memoria che non è soggetta alla paginazione da parte del sistema operativo.   
**---- TODO: profilare con e senza pinned memory e linkare la sezione**     


# Performance e confronto con la versione sequenziale
## specifiche tecniche della macchina utilizzata
## Esempi di test
