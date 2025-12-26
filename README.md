# 🤖 MURMAIL: Multi-Agent Imitation Learning & Emergent Communication

Questo repository contiene l'implementazione di **MURMAIL** (Maximum Uncertainty Response Multi-Agent Imitation Learning) applicato all'ambiente **Simple Speaker Listener** di PettingZoo (MPE).

Il progetto si concentra su due obiettivi principali:
1.  **Emergent Communication:** Forzare la creazione di un protocollo linguistico discreto e robusto tra agenti (A, B, C -> Rosso, Verde, Blu) evitando "cheat" come l'uso del padding.
2.  **Imitation Learning:** Utilizzare MURMAIL per apprendere l'Equilibrio di Nash partendo dalle dimostrazioni di un esperto PPO.

![Language Matrix](discrete_language_matrix_extended.png)
*Figura: Matrice di confusione del linguaggio emerso dopo il training con Signal Jamming. Diagonale perfetta = Comunicazione corretta.*

---

## ✨ Features Principali

### 1. Robust Emergent Communication 🗣️
Per risolvere il problema del "Padding Cheating" (dove lo speaker usava il silenzio per comunicare), è stata implementata una strategia avanzata:
* **Signal Jamming Wrapper:** Un wrapper personalizzato che intercetta le azioni di padding (silenzio) dello speaker e le sostituisce con rumore casuale, rendendo il canale inaffidabile.
* **Entropy Annealing:** Schedulazione lineare dell'entropia (da 0.20 a 0.01) per forzare l'esplorazione iniziale delle parole discrete.
* **Disentanglement:** Risultato finale con mappatura 1-a-1 tra Colori (Target) e Parole (Messaggi).

### 2. MURMAIL Algorithm 🧠
Implementazione dell'algoritmo MURMAIL per giochi a somma zero:
* **Inner Loop RL:** Utilizzo di **UCB-VI** (Upper Confidence Bound Value Iteration) per massimizzare l'incertezza.
* **Synthetic Rewards:** Costruzione di reward basate sull'accordo con la policy dell'esperto.
* **Mirror Descent:** Aggiornamento delle policy tramite Exponentiated Gradient.

### 3. High Performance Engineering ⚡
* **Numba JIT Compilation:** Caricamento e ricostruzione ultra-veloce delle dinamiche di transizione (`fast_loader.py`).
* **WandB Integration:** Logging completo di metriche, loss e gradienti.
* **Automated Pipeline:** Script bash `pipeline.sh` per gestire l'intero ciclo di vita dell'esperimento.

---

## 🚀 Installazione

Assicurati di avere Python 3.10+ installato.

```bash
# Clona il repository
git clone [https://github.com/IL-TUO-USERNAME/MARL.git](https://github.com/IL-TUO-USERNAME/MARL.git)
cd MARL

# Crea un ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
Requisiti principali

pettingzoo[mpe]

stable-baselines3

supersuit

wandb

numba

seaborn

🛠️ Utilizzo (Pipeline)
L'intero esperimento può essere riprodotto eseguendo lo script della pipeline. Questo script gestisce in sequenza: Training dell'esperto, Estrazione dati, Stima dinamiche, Baseline BC e MURMAIL.

Bash
chmod +x pipeline.sh
./pipeline.sh
Step della Pipeline:

Training Esperto (PPO): Addestra lo Speaker-Listener con Signal Jamming per generare un linguaggio perfetto.

Policy Extraction: Converte la rete neurale PPO in policy tabellari per gli stati visitati.

Dynamics Estimation: Stima empiricamente la matrice di transizione P(s 
′
 ∣s,a 
1
​	
 ,a 
2
​	
 ) e reward R.

Behavioral Cloning (Baseline): Esegue una baseline di BC standard.

Run MURMAIL: Esegue l'algoritmo principale.

Confronto: Genera grafici comparativi (comparison_bc_vs_murmail.png).

📂 Struttura del Progetto
Core Algorithms:

murmail.py: Implementazione della classe MaxUncertaintyResponseImitationLearning.

innerloop_rl.py: Implementazione di UCBVI per il loop interno.

behavior_cloning.py: Baseline Multi-Agent Behavioral Cloning.

Training & Environment:

train_ppo.py: Script di training dell'esperto con PPO, Entropy Annealing e Wrapper personalizzati.

env_wrappers.py: Wrapper per visualizzazione e test.

fast_loader.py: Caricamento ottimizzato con Numba.

Analysis:

analyze_language.py: Genera la matrice di confusione del linguaggio (Heatmap).

compare_results.py: Confronta le curve di convergenza (Nash Gap).

extract_expert.py: Estrae le policy dall'agente RL.

📊 Risultati Attesi
Dopo l'esecuzione della pipeline, troverai nella root:

discrete_language_matrix_extended.png: La prova che il linguaggio è emerso correttamente senza cheat.

comparison_bc_vs_murmail.png: Grafico che mostra come MURMAIL riduce l'exploitability (Nash Gap) rispetto a BC.

results_table.tex: Tabella LaTeX con i risultati numerici finali.

👥 Credits
Sviluppato per il corso di Distributed AI - Università di Modena e Reggio Emilia.


### Cosa fare ora:
1.  Crea un file chiamato `README.md` nella cartella principale del tuo progetto.
2.  Incolla il testo qui sopra.
3.  Sostituisci `https://github.com/IL-TUO-USERNAME/MARL.git` con il link vero del tuo repo.
4.  Fai:
    ```bash
    git add README.md
    git commit -m "Add project documentation"
    git push
    ```
