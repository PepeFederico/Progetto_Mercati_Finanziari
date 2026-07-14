# Progetto - Multi-Input LSTM & Econometric Volatility Forecasting
A.A. 2025 - 2026 -- Corso di Studi: Metodi Probabilistici e Statistici per Mercati Finanziari

Questo repository contiene il framework di ricerca e sviluppo dedicato alla modellizzazione e alla previsione della volatilità finanziaria mediante architetture di Deep Learning Deep Ricorsive (**Multi-Input LSTM**) e modelli econometrici classici (**GARCH**).

Il progetto documenta un percorso sperimentale incrementale volto a testare l'Ipotesi di Efficienza dei Mercati (EMH) e a superare i limiti predittivi tradizionali integrando dinamiche multifrequenza e indicatori *forward-looking* di sentiment (**VIX**).

---

## Indice
1. [Evoluzione del Percorso Sperimentale](#1-evoluzione-del-percorso-sperimentale)
2. [Architettura del Modello Finale](#2-architettura-del-modello-finale)
3. [Risultati e Validazione Empirica](#3-risultati-e-validazione-empirica)
4. [Analisi Diagnostica dei Residui](#4-analisi-diagnostica-dei-residui)
5. [Integrazione Futura: Modulo Ibrido AR-LSTM](#5-integrazione-futura-modulo-ibrido-ar-lstm)

---

## 1. Evoluzione del Percorso Sperimentale

La ricerca è stata strutturata secondo un approccio incrementale per isolare il valore aggiunto informativo del Deep Learning applicato alle serie storiche finanziarie:

*   **Fase 1: Predizione Direzionale dei Rendimenti (Singolo Titolo):** Tentativo di prevedere la traiettoria dei rendimenti logaritmici. I test diagnostici hanno confermato la teoria del *Random Walk*: la rete, posta dinanzi al rumore stocastico puro, collassa verso stimatori ingenui (*lagging*) o sulla media.
*   **Fase 2: Volatilità Storica (Singolo Titolo):** Spostamento del target sui rendimenti logaritmici quadrati (volatilità). Pur intercettando la persistenza, i modelli predittivi puri su singoli asset risentono drasticamente di shock informativi esogeni e idiosincratici.
*   **Fase 3: Modello Multi-Input (ETF SXR8.DE + VIX):** Ridefinizione del target sulla volatilità dell'ETF europeo `SXR8.DE` (replica dell'S&P 500) potenziando il set informativo con il canale macroeconomico *forward-looking* dell'indice **VIX**.

---

## 2. Architettura del Modello Finale

Il modello definitivo sfrutta la **Functional API di Keras** per elaborare simultaneamente tre diverse finestre temporali (Multi-Frequency) combinate con l'informazione del VIX:

*   **Ramo Giornaliero:** Lookback di 20 giorni su Asset + VIX $\rightarrow$ LSTM (16 unità)
*   **Ramo Settimanale:** Lookback di 10 giorni su Asset + VIX $\rightarrow$ LSTM (8 unità)
*   **Ramo Trimestrale:** Lookback di 6 giorni su Asset + VIX $\rightarrow$ LSTM (4 unità)

I vettori latenti estratti dalle celle LSTM vengono concatenati e convogliati verso layer densi regolarizzati con *Dropout* e penalizzazione *L2* per prevenire l'overfitting.

---

## 3. Risultati e Validazione Empirica

### A. Capacità Predittiva Out-of-Sample
Il modello ha registrato un **$R^2$ pari al $6.88\%$** sul test set. Sebbene quantitativamente contenuto, il risultato è statisticamente significativo e coerente con la natura fortemente stocastica della varianza nei mercati finanziari.

### B. Analisi di Varianza-Covarianza ed Effetto Smoothing
L'analisi della matrice di varianza-covarianza tra volatilità reale ($y$) e prevista ($\hat{y}$) evidenzia un rapporto percentuale delle varianze $[\text{Var}(\hat{y}) / \text{Var}(y)]$ pari al **$6.91\%$**. 
Questo dato quantifica matematicamente l'effetto di *smoothing* terapeutico della rete: l'algoritmo isola il trend strutturale di lungo periodo filtrando il 93% del rumore microscopico *intraday*.

### C. Benchmark Econometrico: GARCH(1,1) vs VIX
Il confronto tra la volatilità condizionale statistica (GARCH) e l'indice VIX ha mostrato una **Correlazione di Pearson del $65.84\%$** ($p\text{-value} \approx 0$). 
La divergenza riscontrata in specifici periodi (es. inizio 2026) evidenzia il **Volatility Risk Premium** (il premio per il rischio/panico futuro incorporato nel VIX), giustificando scientificamente la scelta di inserire il VIX come feature esogena per colmare i limiti *backward-looking* del GARCH.

---

## 4. Analisi Diagnostica dei Residui

### Il Significato del Lag +6 nella Cross-Correlazione (CCF)
Lo studio della Funzione di Cross-Correlazione ha evidenziato il picco massimo di significatività in corrispondenza del **Lag +6** (circa una settimana di borsa aperta). 

*   **Cosa NON significa:** Non indica che il modello è in ritardo o che prevede lo shock dopo una settimana. Il modello reagisce istantaneamente al tempo $t$ grazie agli impulsi del VIX.
*   **Cosa SIGNIFICA operativamente:** Indica che la predizione formulata oggi dalla LSTM ha la massima capacità esplicativa nei confronti della **scia di instabilità e persistenza** che lo shock riversa sul mercato nei 6 giorni successivi. Si configura quindi come un eccellente *previsore di regime strutturale* a medio termine.
*   **I Lag Negativi:** La totale assenza di significatività statistica nei lag negativi certifica che il modello non soffre di *lagging passivo* (non rincorre i dati passati).

---

## 5. Integrazione Futura: Modulo Ibrido AR-LSTM

Per sanare la memoria lineare a brevissimo termine emersa al Lag 1 nel test di Ljung-Box e mitigare il ritardo di fase, il repository predispone l'architettura per un'estensione ibrida **Linear + Non-Linear** inserendo un filtro **AR(1)** in parallelo alle LSTM:
