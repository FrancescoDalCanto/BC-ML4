# PIANO DI LAVORO TESI - PRIORITÀ E TIMELINE

## 🚨 PRIORITÀ IMMEDIATA (DA FARE SUBITO)

### 1. Email a Gianluca ⚡
**URGENTE - Prima di tutto**
- Chiedere se i file forniti sono segmentazioni **originali** o **preprocessate**
- Da questo dipendono molte modifiche alla tesi

**Email suggerita:**
```
Oggetto: Chiarimento segmentazioni per tesi

Ciao Gianluca,

per procedere con le modifiche alla tesi ho bisogno di un chiarimento:
i file di segmentazione che mi hai fornito (quelli senza "MedSum") 
sono segmentazioni originali o già preprocessate?

Grazie mille!
```

---

## 📊 FASE 1: RICALCOLI E ANALISI (1-2 settimane)

### Task 1.1: Setup delle segmentazioni
**Tempo stimato: 1 giorno**
- [ok] Ricevere risposta da Gianluca
- [ok] Verificare quali file usare (originali vs preprocessate)
- [ok] Preparare due set di dati:
  - Segmentazioni originali → Radiomic Lesion
  - Segmentazioni MedSum → Radiomic MedSum
- [ok] Testare che i file si carichino correttamente

### Task 1.2: Ricalcolo risultati su nuove segmentazioni
**Tempo stimato: 2-3 giorni**
- [ok] Modificare il codice per caricare il nuovo file all'inizio
- [ok] Ricalcolare TUTTI i risultati con Radiomic_MedSum (Radiomic Lesion)
- [ok] Per tutti i modelli (XGBoost, Random Forest, SVM, Logistic Regression)
- [ok] Per tutte le configurazioni (1, 2, 3)
- [ok] Per tutti i target (Estrogeni, Progesterone, HER2)
- [ok] Salvare risultati in formato tabellare

**Deliverable:** Nuove tabelle risultati

### Task 1.3: Cross-validation tra dataset
**Tempo stimato: 2 giorni**
- [ok] Implementare cross-validation MBL → Duke
- [ok] Implementare cross-validation Duke → MBL
- [ok] Solo per **estrogeni**
- [ok] NON usare il CSV mergiato
- [ok] Per tutti i modelli (o almeno XGBoost)
- [ok] Salvare risultati

**Deliverable:** Tabelle cross-validation

### Task 1.4: Feature Importance
**Tempo stimato: 3-4 giorni**
- [ ] Calcolare Feature Importance per XGBoost su estrogeni
- [ ] Calcolare per entrambi i dataset separati
- [ ] Calcolare per dataset mergiato (opzionale, da vedere)
- [ ] Estrarre top 10-15 features per ogni configurazione
- [ ] Creare tabella di confronto per vedere sovrapposizioni
- [ ] Eventualmente calcolare per altri modelli che performano bene

**Deliverable:** Tabelle Feature Importance + analisi sovrapposizioni

### Task 1.5: Boxplot (se hai tempo)
**Tempo stimato: 1 giorno**
- [ ] Salvare risultati di tutti i fold (non solo media/std)
- [ ] Creare boxplot usando i valori individuali
- [ ] Verificare se vengono meglio rispetto alla versione multiclasse
- [ ] Se vengono male, lasciar perdere

**Deliverable:** Grafici boxplot (opzionali)

---

## 📝 FASE 2: MODIFICHE ALLA TESI (1 settimana)

### Task 2.1: Sezione Dataset
**Tempo stimato: 2 ore**
- [ ] Verificare che ci siano solo MBL e Duke (niente merge qui)
- [ ] In base a risposta Gianluca:
  - Se preprocessate → lasciare spiegazione preprocessing
  - Se originali → commentare paragrafo preprocessing
- [ ] Commentare tutta la parte sulle dinamiche

### Task 2.2: Sezione Materiali e Metodi - Ristrutturazione
**Tempo stimato: 3-4 ore**
- [ ] Spostare "Iperparametri" vicino a "Algoritmi"
- [ ] Rinominare sezione: "Gestione sbilanciamento" → "Configurazioni sperimentali per la gestione dei dati"
- [ ] Aggiungere breve frase sul merge dataset
- [ ] Eventualmente aggiungere tabella: dati prima e dopo merge

### Task 2.3: Sezione Explainability (NUOVA)
**Tempo stimato: 3 ore**
- [ ] Cercare info su Feature Importance
- [ ] Scrivere paragrafo breve e generale (max 1 pagina)
- [ ] Posizionare alla fine di Materiali e Metodi
- [ ] Spiegare cos'è e perché è utile

### Task 2.4: Sezione Risultati - Tabelle
**Tempo stimato: 1 giorno**
- [ ] Riorganizzare ordine:
  1. Duke MedSum
  2. Duke Originali
  3. MBL MedSum
  4. MBL Originali
  5. Mergiato
- [ ] Creare tabelle con solo F1, Accuracy, AUC
- [ ] Aggiungere deviazione standard sotto ogni valore (es: 0.85 ± 0.03)
- [ ] Creare 2 tabelle comparative (MedSum vs Originali)
  - 4 righe (algoritmi) × 3 colonne (metriche)
- [ ] Mantenere capitolo confronto modelli (per dire XGBoost è migliore)

### Task 2.5: Sezione Risultati - Feature Importance
**Tempo stimato: 4-5 ore**
- [ ] Inserire tabelle Feature Importance dopo risultati performance
- [ ] Decidere quante features mostrare (10-15-20?)
- [ ] Creare visualizzazioni se necessario
- [ ] Commentare le sovrapposizioni trovate

### Task 2.6: Grafici
**Tempo stimato: 1 giorno**
- [ ] Per ogni tabella risultati:
  - Curva ROC del **migliore** risultato
  - Curva ROC del **peggiore** risultato
- [ ] Verificare che le curve abbiano forme normali
- [ ] Se boxplot sono venuti bene, inserirli

### Task 2.7: Sezione Conclusioni - Ristrutturazione COMPLETA
**Tempo stimato: 4-6 ore**

**STRUTTURA NUOVA:**

#### 1. Discussione dei Risultati (UN SOLO PARAGRAFO)
- [ ] Riassunto: cosa hai fatto nella tesi
- [ ] Problema affrontato: scelta classificatore ML
- [ ] Risultato: XGBoost performa meglio
- [ ] Inserire NUMERI dei risultati migliori
- [ ] Osservazione chiave: funziona **solo su estrogeni**
- [ ] Interpretazione: feature radiomiche esprimono bene estrogeni, meno progesterone/HER2
- [ ] Analisi Feature Importance: cosa hai scoperto

#### 2. Limitazioni (PARAGRAFO SEPARATO)
- [ ] Scrivere limitazioni del lavoro
- [ ] Farlo leggere alla relatrice per approvazione

#### 3. Sviluppi Futuri
- [ ] Cosa si potrebbe fare per migliorare
- [ ] Decidere se unirlo a Limitazioni o Conclusioni Finali

#### 4. Conclusioni Finali (3-4 FRASI)
- [ ] In questa tesi ho fatto X
- [ ] Ho ottenuto buone performance [numero]
- [ ] Frase finale di chiusura

---

## 🎨 FASE 3: RIFINITURA FINALE (2-3 giorni)

### Task 3.1: Revisione completa
**Tempo stimato: 1 giorno**
- [ ] Rileggere tutta la tesi dall'inizio
- [ ] Verificare coerenza tra sezioni
- [ ] Controllare che tutto ciò che è nell'introduzione sia nelle conclusioni
- [ ] Verificare numerazione tabelle/figure
- [ ] Verificare riferimenti incrociati

### Task 3.2: Formattazione
**Tempo stimato: 3-4 ore**
- [ ] Uniformare formato tabelle
- [ ] Uniformare formato grafici
- [ ] Verificare intestazioni e didascalie
- [ ] Controllare bibliografia

### Task 3.3: Ringraziamenti
**Tempo stimato: 1-2 ore**
- [ ] **ULTIMA COSA IN ASSOLUTO**
- [ ] Scrivere ringraziamenti
- [ ] Rileggere a mente fresca

---

## 📅 TIMELINE SUGGERITA

### Settimana 1
- **Giorno 1:** Email Gianluca + Setup segmentazioni
- **Giorni 2-4:** Ricalcolo tutti i risultati
- **Giorni 5-6:** Cross-validation
- **Giorno 7:** Inizio Feature Importance

### Settimana 2
- **Giorni 1-3:** Completare Feature Importance
- **Giorno 4:** Boxplot (se tempo)
- **Giorni 5-7:** Modifiche sezioni Dataset, Metodi, Explainability

### Settimana 3
- **Giorni 1-3:** Tabelle e grafici risultati
- **Giorni 4-5:** Ristrutturazione Conclusioni
- **Giorni 6-7:** Revisione e rifinitura

### Settimana 4 (buffer)
- Tempo extra per imprevisti
- Ultima revisione
- Ringraziamenti

---

## 📋 CHECKLIST VELOCE GIORNALIERA

**Ogni giorno prima di iniziare:**
- [ ] Cosa devo fare oggi?
- [ ] Ho tutti i file/dati necessari?
- [ ] Ho fatto backup del lavoro di ieri?

**Ogni sera prima di finire:**
- [ ] Cosa ho completato oggi?
- [ ] Cosa rimane da fare domani?
- [ ] Ho salvato tutto su cloud/Git?
- [ ] Ho fatto commit del codice?

---

## ⚠️ NOTE IMPORTANTI

1. **NON CANCELLARE MAI NIENTE** → Commentare sempre il codice/testo
2. **Salvare versioni intermedie** della tesi (es: tesi_v1.pdf, tesi_v2.pdf)
3. **Backup giornaliero** su cloud (Google Drive, Dropbox, etc.)
4. **Tenere log** dei risultati ottenuti giorno per giorno
5. **Se qualcosa non funziona** → annotare il problema e andare avanti, tornarci dopo

---

## 🎯 OBIETTIVI DI CONSEGNA

- [ ] Tutti i calcoli completati
- [ ] Tutte le tabelle aggiornate
- [ ] Tutti i grafici inseriti
- [ ] Tesi ristrutturata secondo indicazioni
- [ ] Conclusioni complete e coerenti
- [ ] Tesi revisionata
- [ ] Ringraziamenti scritti
- [ ] **PRONTA PER INVIO ALLA RELATRICE**

---

## 💡 CONSIGLI

- Inizia dalle cose che ti bloccano (email a Gianluca)
- Se un task è troppo grande, spezzalo in sotto-task
- Fai una cosa alla volta, completa, poi passa alla successiva
- Non cercare la perfezione al primo colpo
- Chiedi feedback alla relatrice quando hai completato una fase

**Buon lavoro! 💪**