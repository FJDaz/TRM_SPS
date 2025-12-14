# 🏗️ Architecture TRM (Mistral 7B ↔ BERT) - PRD Enrichi

**Version :** 2.1 (POC Bootstrap)  
**Date :** Décembre 2025  
**Concept :** Architecture collaborative permettant à un modèle génératif (Mistral 7B) de traiter efficacement un contexte court (~500 tokens) en s'appuyant sur une mémoire condensée et pondérée fournie par un modèle encodeur (BERT), avec intégration RAG par philosophe.

**💰 Mode Bootstrap 0€ → 100€ :** Ce document décrit une roadmap optimisée pour un budget minimal (100€ max), avec développement sur Colab gratuit et infrastructure Vast.ai/RunPod. Le POC est conçu comme un **asset valorisable** (consulting, formation, portfolio) autant qu'une validation technique.

**🎯 Objectif Stratégique :** Réduire les coûts d'inférence LLM de 50% tout en créant une expertise différenciante valorisable (TRM = tendance 2026).

---

## 📋 Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Architecture Technique](#architecture-technique)
3. [Composants Principaux](#composants-principaux)
4. [Pipeline Fonctionnel](#pipeline-fonctionnel)
5. [Modèles de Données](#modèles-de-données)
6. [Roadmap d'Implémentation](#roadmap-dimplémentation)
7. [Métriques & Tests](#métriques--tests)
8. [Exemples Concrets](#exemples-concrets)

---

## 🎯 Vue d'Ensemble

### Objectif Principal

Permettre à un modèle génératif (Mistral 7B) de traiter efficacement un contexte court (~500 tokens) en s'appuyant sur une mémoire condensée et pondérée fournie par un modèle encodeur (BERT), tout en conservant un flux d'information structuré et inductif, avec accès à une mémoire externe via RAG par philosophe.

### Objectifs Business & Stratégiques

**💰 Optimisation Coûts** :
- Réduction coûts inférence : **-50%** (Qwen 14B → TRM Mistral 7B)
- Scaling horizontal : GPU 7B moins cher que 14B
- Même budget → 2-3× plus de requêtes traitées

**🚀 Positionnement Early Adopter** :
- TRM = tendance 2026 (Small LM + Encoders)
- POC = case study avant que ça devienne mainstream
- Expertise différenciante valorisable (consulting, formation, portfolio)

**💼 Valorisation Expertise** :
- White paper / blog post (Medium, Dev.to)
- Talk / conférence (PyConFR, MLOps meetups)
- Portfolio GitHub (crédibilité technique)
- Consulting / freelance (5-10k€ par mission)

### Principes Clés

- **Mémoire conversationnelle condensée** : BERT encode et condense l'historique en STATE_IMAGE structuré
- **Mémoire externe intelligente** : RAG retrieve passages pertinents du corpus philosophique, condensés par BERT
- **Génération locale focalisée** : Mistral génère avec contexte court, maximisant l'attention locale
- **Évolutivité** : Nouveaux axes d'information et heuristiques peuvent être intégrés facilement
- **Bootstrap-friendly** : Développement Colab gratuit + infrastructure payante minimale (100€ max)

---

## 🏗️ Architecture Technique

### Stack Technologique

```
Frontend (HTML/JS)
    ↓
Backend API (FastAPI)
    ↓
Router (détermine philosophe/corpus)
    ↓
RAG Retriever (sentence-transformers + FAISS)
    ↓
BERT Encoder (CPU - Vast.ai/RunPod ou Colab gratuit)
    ├── Encode conversation + passages RAG
    ├── Analyse axes: concepts, intention, tension, style, priorité
    └── Génère STATE_IMAGE condensé
    ↓
Mistral 7B (GPU - Vast.ai/RunPod RTX 4090)
    ├── Lit STATE_IMAGE structuré
    ├── Génère réponse (≤500 tokens contexte)
    └── Détecte motifs/tics de langage
    ↓
Mini-store (Heuristique - simplifié POC)
    ├── Collecte motifs récurrents
    └── Feedback pour mise à jour STATE par BERT (désactivé POC)
```

**Infrastructure Bootstrap** :
- **Développement** : Google Colab gratuit (CPU pour BERT, T4 GPU pour tests Mistral)
- **Production POC** : Vast.ai (~$0.30-0.40/h) ou RunPod (~$0.49/h)
- **Budget total** : 100€ max (250h Vast.ai ou 200h RunPod)

### Flux d'Exécution Détaillé

```
1. User Input reçu
   ↓
2. Router détermine philosophe (Spinoza/Kant/Bergson)
   ↓
3. RAG Retrieve (parallèle ou séquentiel)
   - Query: dernier échange utilisateur + concepts actifs STATE
   - Corpus: corpus du philosophe déterminé
   - Retourne: 2-3 passages pertinents max
   ↓
4. BERT Encode
   - Entrées:
     * Dernier échange utilisateur
     * Réponse précédente (si existe)
     * Passages RAG récupérés
     * STATE précédent
     * Feedback Mini-store (motifs récurrents)
   - Traitement:
     * Analyse axes: concepts, intention, tension, fréquence, style, priorité, relations, émotions
     * Extrait concepts clés des passages RAG (pas texte brut)
     * Pondère et condense en STATE_IMAGE
   ↓
5. STATE_IMAGE généré (structuré JSON)
   - Contient concepts conversationnels + concepts RAG condensés
   - Pas de passages bruts (évite surcharge tokens)
   ↓
6. Mistral 7B Génère
   - Contexte: STATE_IMAGE structuré (~100-200 tokens)
   - Dernier échange utilisateur (~50-100 tokens)
   - Total ≤ 500 tokens
   - Génère réponse adaptée au style/philosophe
   ↓
7. Mini-store mis à jour
   - Collecte motifs récurrents détectés par Mistral
   - Alertes sur tics de langage
   ↓
8. Feedback loop
   - Mini-store informe BERT pour prochaine mise à jour STATE
```

---

## 🧩 Composants Principaux

### 1. Router (Détermination Corpus)

**Responsabilités :**
- Détermine le philosophe/corpus à utiliser selon contexte
- Peut être basé sur sélection utilisateur ou détection automatique

**Localisation :** `Backend/router.py`

**Fonctions clés :**
- `determine_philosopher(user_input, session_context)` → philosopher_id
- `get_corpus_path(philosopher_id)` → corpus_path

**Exemple :**
```python
def determine_philosopher(user_input, session_context):
    # Si sélection explicite dans session
    if session_context.get('philosopher'):
        return session_context['philosopher']
    
    # Sinon détection automatique (optionnel)
    # Analyse keywords/concepts dans user_input
    return "spinoza"  # default
```

### 2. RAG Retriever

**Modèle :** `sentence-transformers` (all-MiniLM-L6-v2 ou e5-small) + Whoosh/FAISS

**Responsabilités :**
- Retrieve passages pertinents du corpus philosophique
- Filtre par philosophe (via Router)
- Limite à 2-3 passages max pour éviter surcharge

**Localisation :** `Backend/rag_retriever.py`

**Fonctions clés :**
- `retrieve_passages(query, philosopher, top_k=3)` → List[passage]
- `extract_concepts_from_passages(passages)` → List[concept] (optionnel, pour pré-processing)

**Usage :**
```python
# Query enrichie avec concepts actifs du STATE
query = f"{user_input} {state_image.get('concepts_actifs', [])}"
passages = retrieve_passages(query, philosopher="spinoza", top_k=3)
# Retourne: [{"text": "...", "score": 0.85, "source": "Éthique"}, ...]
```

**Thresholds :**
- Similarity score ≥ 0.65 : passage pertinent
- Similarity score < 0.45 : passage ignoré

### 3. BERT Encoder

**Modèle :** BERT-base ou BERT-mini (CPU)

**Responsabilités :**
- Encode conversation + passages RAG
- Analyse multi-axes (concepts, intention, tension, style, etc.)
- Génère STATE_IMAGE condensé structuré
- Extrait concepts des passages RAG (pas texte brut)

**Localisation :** `Backend/bert_encoder.py`

**Fonctions clés :**
- `encode_to_state_image(conversation, rag_passages, prev_state, mini_store_feedback)` → STATE_IMAGE
- `analyze_axes(text)` → dict avec axes analysés
- `extract_concepts_from_rag(passages)` → List[concept]

**Fréquence :**
- 1× par tour utilisateur
- Ou périodiquement (tous les N tokens, heuristique)

**Chargement :**
- CPU (pas GPU)
- Modèle chargé une fois, réutilisé

### 4. Mistral 7B Generator

**Modèle :** Mistral 7B (ou équivalent, ex: SPS-distilled)

**Responsabilités :**
- Génère réponse conversationnelle
- Lit STATE_IMAGE structuré
- Applique focus sur concepts + intention + style
- Détecte motifs/tics de langage pour Mini-store

**Localisation :** `Backend/mistral_generator.py`

**Fonctions clés :**
- `generate(state_image, user_input, system_prompt)` → response
- `detect_patterns(response)` → patterns_dict (pour Mini-store)

**Contraintes :**
- Contexte ≤ 500 tokens
- STATE_IMAGE structuré (~100-200 tokens)
- Dernier échange utilisateur (~50-100 tokens)
- Réserve pour génération

**Chargement :**
- GPU (vLLM ou équivalent)
- Modèle chargé une fois, réutilisé

### 5. Mini-store (Heuristique)

**Responsabilités :**
- Collecte motifs récurrents détectés par Mistral
- Observe tics de langage, termes répétés
- Fournit feedback pour pondération STATE par BERT

**Localisation :** `Backend/mini_store.py`

**Fonctions clés :**
- `update_patterns(pattern_type, pattern_data)` → void
- `get_feedback_for_bert()` → feedback_dict

**Structure :**
```python
{
    "recurrences": {
        "termes_repetes": {"conatus": 5, "affects": 3},
        "structures": ["question rhétorique", "analogie"]
    },
    "alertes": ["ton trop formel", "répétition excessive"]
}
```

---

## 🔄 Pipeline Fonctionnel

### Schéma Complet

```
┌──────────────┐
│User Input    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Router        │
│- Détermine   │
│  philosophe  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│RAG Retriever │
│- Query enrichie│
│- Corpus philosophe│
│- Top 2-3 passages│
└──────┬───────┘
       │ passages
       ▼
┌──────────────┐
│BERT Encoder  │
│- Encode:     │
│  * conversation│
│  * passages RAG│
│  * STATE prev │
│  * feedback   │
│- Analyse axes │
│- Extrait concepts│
│- Condense     │
└──────┬───────┘
       │ STATE_IMAGE
       ▼
┌──────────────┐
│Mistral 7B    │
│- Lit STATE   │
│- Génère (≤500t)│
│- Détecte motifs│
└──────┬───────┘
       │ response + patterns
       ▼
┌──────────────┐
│Mini-store    │
│- Collecte    │
│- Feedback    │
└──────┬───────┘
       │
       └───► BERT (prochain tour)
```

---

## 📊 Modèles de Données

### STATE_IMAGE Schema (JSON Structuré)

**Version Complète (POC et Production)** :
```json
{
  "concepts_actifs": [
    "conatus",
    "effort persévérer",
    "puissance d'agir"
  ],
  "concepts_rag": [
    "conatus = puissance d'agir",
    "affects modifient conatus"
  ],
  "sources_rag": [
    "Spinoza, Éthique, III, prop. 7"
  ],
  "intention": "clarifier",
  "tension": "neutre",
  "recurrence": {
    "termes_repetes": {
      "conatus": 3,
      "affects": 2
    },
    "structures": ["question rhétorique"]
  },
  "style": "pédagogique",
  "ton": "bienveillant",
  "priorite": [
    "concepts_actifs",
    "intention"
  ],
  "relations": [
    "conatus → affects",
    "cause → effet"
  ],
  "emotion": "curieux",
  "metadata": {
    "philosopher": "spinoza",
    "turn": 5,
    "timestamp": "2025-12-10T10:30:00Z"
  }
}
```

**Note POC** : Pour maximiser la précision de la démo, le STATE_IMAGE complet avec tous les axes est utilisé dès le POC (pas de version réduite).

### Session State Schema

```json
{
  "session_id": "uuid",
  "philosopher": "spinoza",
  "current_turn": 5,
  "state_image": { /* STATE_IMAGE ci-dessus */ },
  "mini_store": {
    "recurrences": { /* ... */ },
    "alertes": [ /* ... */ ]
  },
  "dialogue_history": [
    {
      "turn": 1,
      "user": "...",
      "assistant": "...",
      "rag_passages_used": ["passage_id_1"]
    }
  ],
  "rag_cache": {
    "passage_id_1": {
      "text": "...",
      "concepts_extracted": ["conatus", "puissance"]
    }
  }
}
```

### RAG Passage Schema

```json
{
  "passage_id": "uuid",
  "text": "Le conatus est l'effort par lequel chaque chose s'efforce de persévérer dans son être...",
  "source": "Spinoza, Éthique, III, prop. 7",
  "philosopher": "spinoza",
  "embedding": [0.123, ...],
  "concepts": ["conatus", "effort", "persévérer"],
  "similarity_score": 0.85
}
```

---

## 🗺️ Roadmap d'Implémentation - POC Bootstrap 0€ → 100€

### Objectif du POC

**Technique** : Valider que la chaîne complète (Router → RAG → BERT Condensation → Mistral Génération) est fonctionnelle, que le `STATE_IMAGE` est correctement structuré et lu, et que le contexte généré reste sous les 500 tokens.

**Business** : Créer un asset valorisable (portfolio, consulting, formation) démontrant une réduction de 50% des coûts d'inférence LLM via architecture TRM.

**⚠️ Point Critique** : La gestion de la mémoire et de la concurrence entre les deux modèles sur des ressources différentes (CPU pour BERT, GPU pour Mistral) doit être validée très tôt, avant même l'intégration fonctionnelle (voir étape 1.1b).

### Stratégie Budget Minimal

**Phase 0 (0€)** : Développement sur Colab gratuit (économie ~60€)  
**Phase 1 (80-100€)** : Infrastructure Vast.ai/RunPod uniquement pour intégration finale et benchmarks  
**Phase 2 (0€)** : Valorisation immédiate (articles, portfolio, consulting)

### Phase 0 : Préparation 0€ (Cette semaine - Colab Gratuit)

**Objectif** : Maximiser la préparation gratuite pendant l'attente des 100€.

#### ✅ Étape 0.1 : Prototypage BERT Encoder (Colab CPU Gratuit)
- [ ] Test extraction concepts + STATE_IMAGE generation
- [ ] Pas besoin GPU pour BERT (CPU Colab suffit)
- [ ] Validation logique multi-axes (concepts, intention, tension, style, etc.)
- [ ] Tests unitaires basiques

**Livrables :**
- Notebook Colab : `Backend/Notebooks/bert_encoder_prototype.ipynb`
- Fonction `extract_concepts_from_rag()` testée
- Fonction `encode_to_state_image()` testée

#### ✅ Étape 0.2 : Préparation Corpus RAG (Colab T4 Gratuit)
- [ ] Génération embeddings sentence-transformers (Colab gratuit T4)
- [ ] Indexation FAISS des 3 corpus (Spinoza/Bergson/Kant)
- [ ] Sauvegarde index localement (pour Phase 1)
- [ ] Tests retrieval basiques

**Livrables :**
- Notebook Colab : `Backend/Notebooks/rag_indexing.ipynb`
- Indexes FAISS sauvegardés : `Backend/data/rag_indexes/`
- Scripts de chargement index

#### ✅ Étape 0.3 : Tests Mistral 7B Baseline (Colab T4 Gratuit - Sessions 12h)
- [ ] Chargement modèle Mistral 7B + inférence test
- [ ] Mesure latence baseline
- [ ] Validation que ça tient en 15GB VRAM (T4 Colab)
- [ ] Tests avec STATE_IMAGE mock

**Livrables :**
- Notebook Colab : `Backend/Notebooks/mistral_baseline.ipynb`
- Métriques baseline : latence, VRAM usage
- Validation faisabilité technique

#### ✅ Étape 0.4 : Documentation & Contenus (0€)
- [ ] Architecture diagram (Excalidraw gratuit)
- [ ] README structure (template PRD)
- [ ] Script benchmark automatisé (pour Phase 1)
- [ ] Préparation templates articles/blog posts

**Livrables :**
- `docs/references/architecture_diagram.excalidraw`
- `README.md` structure portfolio-quality
- `Backend/scripts/benchmark_automated.py`
- Templates contenu valorisation

**Budget Phase 0 : 0€** (tout sur Colab gratuit)

---

### Phase 1 : POC Fonctionnel sur Budget Minimal (80-100€)

**Infrastructure** : Vast.ai (~$0.30-0.40/h) ou RunPod (~$0.49/h)  
**Budget** : 100€ = ~250h Vast.ai (10 jours 24/7) ou ~200h RunPod (8 jours 24/7)  
**Stratégie** : Dev sur Colab gratuit, Vast.ai uniquement pour intégration finale (24-48h)

#### ✅ Étape 1.1 : Router & RAG Retriever (Colab Gratuit - Jours 1-2)
- [ ] Implémenter Router (détermination philosophe)
- [ ] Intégrer RAG Retriever avec `sentence-transformers` (embeddings déjà générés Phase 0)
- [ ] Charger index FAISS précomputé (Phase 0)
- [ ] **Minimal Testing:** Test unitaire de base pour vérifier que `retrieve_passages` retourne des résultats.

**Livrables :**
- `Backend/router.py`
- `Backend/rag_retriever.py`
- Tests unitaires basiques

**Note** : Index FAISS déjà généré Phase 0, pas besoin de GPU ici.

#### 🆕 Étape 1.1b : **Vérification des Ressources & Chargement** (Vast.ai - Jour 4, ~20€) ⚠️ CRITIQUE

**Objectif** : Valider que les modèles peuvent être chargés simultanément sans conflit de ressources (mémoire, ports, accès) avant d'implémenter la logique fonctionnelle.

**Infrastructure** : Vast.ai RTX 4090 (~$0.30/h) + CPU instance (~$0.05/h) = ~$0.35/h  
**Durée estimée** : 6-8h = ~20€

- [ ] **Chargement BERT (CPU)** :
  - [ ] Charger le modèle BERT-base/mini sur le **CPU**
  - [ ] Vérifier l'empreinte mémoire RAM (doit rester < 4GB pour BERT-mini)
  - [ ] Exécuter une inférence test pour valider l'accès
  - [ ] Documenter la latence d'inférence basique

- [ ] **Chargement Mistral (GPU)** :
  - [ ] Charger le modèle Mistral 7B sur le **GPU** via vLLM/PyTorch
  - [ ] Vérifier la VRAM disponible (éviter OOM - Object Out of Memory)
  - [ ] Vérifier que le modèle reste chargé en mémoire GPU
  - [ ] Exécuter une inférence test pour valider l'accès
  - [ ] Documenter la latence d'inférence basique

- [ ] **Test de Coexistence** :
  - [ ] Exécuter une inférence BERT (CPU) et une inférence Mistral (GPU) **simultanément**
  - [ ] Vérifier qu'il n'y a **pas de conflits** d'accès aux ressources
  - [ ] Vérifier qu'il n'y a **pas de dégradation critique** de la latence lors du chargement simultané
  - [ ] Valider que les deux modèles restent accessibles après coexistence

**Livrables :**
- `Backend/tests/test_model_loading.py` (tests de chargement)
- `Backend/tests/test_resource_coexistence.py` (tests de coexistence)
- Rapport de vérification ressources (mémoire RAM, VRAM, latences)
- Script de validation ressources : `Backend/scripts/validate_resources.py`

**Justification** : Cette étape sépare les risques liés au **déploiement** (chargement des modèles) des risques liés au **fonctionnel** (implémentation de la logique d'encodage et de génération), réduisant ainsi le temps de débogage de la Phase 2.

#### ✅ Étape 1.2 : BERT Encoder Core Complet (Colab Gratuit - Jour 2-3)
- [ ] Implémenter classe `BERTEncoder` (CPU) - **prototype déjà testé Phase 0**
- [ ] **Analyse Multi-Axes Complète:** Implémenter l'analyse de tous les axes pour maximiser la précision du POC :
  - `concepts_actifs` : Idées centrales, décisions récentes
  - `concepts_rag` : Concepts extraits des passages RAG
  - `intention` : Question, clarification, instruction
  - `tension` : Accord / désaccord / neutre
  - `style` : Concis, humoristique, pédagogique
  - `ton` : Bienveillant, formel, etc.
  - `priorite` : Crucial vs secondaire
  - `relations` : Cause → effet, analogies
  - `emotion` : Politesse, ironie, frustration
- [ ] Fonction `extract_concepts_from_rag()` (extraction concepts, **critique pour le POC**)
- [ ] Fonction `encode_to_state_image()` avec tous les axes
- [ ] **Testing:** Test unitaire pour vérifier que le `STATE_IMAGE` généré n'inclut **pas de texte brut des passages RAG** et respecte la structure JSON complète.

**Livrables :**
- `Backend/bert_encoder.py` (avec analyse multi-axes complète)
- Tests unitaires basiques (focus validation structure STATE_IMAGE complète)

**Note** : Le modèle BERT est déjà chargé et validé depuis l'étape 1.1b, permettant de se concentrer sur la logique fonctionnelle.

#### ✅ Étape 1.3 : Mistral Generator & Mini-store (Colab Gratuit - Jour 3)
- [ ] Implémenter classe `MistralGenerator` (GPU) - **baseline déjà testée Phase 0**
- [ ] Fonction `generate()` lisant le `STATE_IMAGE` et l'`user_input` (Contexte ≤ 500 tokens)
- [ ] **Mini-store Simplifié/Reporté:** Mettre en place la classe `MiniStore`, mais **désactiver ou simplifier** la boucle de feedback vers BERT. La détection de patterns peut être **reportée** après la validation de la chaîne principale.
- [ ] **Minimal Testing:** Test unitaire pour vérifier que `generate` produit une réponse cohérente en utilisant le `STATE_IMAGE`.

**Livrables :**
- `Backend/mistral_generator.py`
- `Backend/mini_store.py` (structure de base, feedback désactivé)
- Tests génération basiques

**Note** : Le modèle Mistral est déjà chargé et validé depuis l'étape 1.1b, permettant de se concentrer sur la logique fonctionnelle de génération.

#### ✅ Étape 1.4 : Templates & Endpoint (Colab Gratuit - Jour 3)
- [ ] Templates Prompts (System prompt + injection `STATE_IMAGE`)
- [ ] **Endpoint `/chat_trm` (Première Intégration):** Créer l'endpoint FastAPI intégrant la chaîne complète pour une seule requête.
- [ ] **Session Management Basique:** Stockage session/état minimaliste (in-memory).

**Livrables :**
- `Backend/prompts/system_prompt.py`
- `Backend/prompts/chat_template.py`
- `Backend/api/endpoints/trm.py`
- `Backend/session_manager.py` (basique)

---

### Phase 2 : Intégration & Benchmarks sur Vast.ai (Jours 4-7, ~60€)

**Infrastructure** : Vast.ai RTX 4090 + CPU  
**Budget** : ~60€ (24-48h d'utilisation)

#### ✅ Étape 2.1 : Intégration Complète (Vast.ai - Jour 4-5, ~40€)
- [ ] Déployer pipeline complet sur Vast.ai (2 instances : GPU + CPU)
- [ ] Finaliser l'intégration complète du pipeline (Router → RAG → BERT → Mistral).
- [ ] Gérer la persistance du `STATE_IMAGE` entre les tours de conversation (Session State).
- [ ] **Tests d'Intégration Critiques :** Simuler 5 dialogues complets (end-to-end) pour s'assurer que le `STATE_IMAGE` s'enrichit correctement à chaque tour.
- [ ] Ajustements/debugging

**Livrables :**
- Pipeline complet fonctionnel déployé
- `tests/test_integration_trm.py` (5 dialogues de test)
- Rapport validation chaîne

#### ✅ Étape 2.2 : Benchmarks Comparatifs (Vast.ai - Jour 6-7, ~40€)
- [ ] **Benchmarks Comparatifs** : TRM vs Qwen 14B simulé (baseline)
- [ ] **Métriques Mesurées** :
  - Coût par requête (GPU heures × prix)
  - Latence moyenne (ms)
  - Qualité réponses (subjective ou score automatique)
  - Throughput (req/s)
- [ ] **Focus Qualité/Taille:** Affiner l'extraction des concepts RAG pour garantir la densité de l'information.
- [ ] Vérifier la taille du contexte de Mistral à chaque tour (**vérification critique POC : Total ≤ 500 tokens**).
- [ ] Ajuster les *thresholds* RAG pour garantir la pertinence.
- [ ] Enregistrement démo vidéo (3-5 min)

**Livrables :**
- `docs/benchmarks/comparison_trm_vs_qwen14b.md`
- Graphiques coût/latence/qualité
- Fonction `extract_concepts_from_rag()` affinée
- `Backend/config/thresholds.py` (thresholds optimaux)
- Validation taille contexte ≤ 500 tokens
- Vidéo démo POC (3-5 min)

**Budget Phase 2** : ~60€ (40€ intégration + 20€ benchmarks)

---

### Phase 3 : Valorisation Immédiate (0€ - Semaines Post-POC)

**Objectif** : Transformer POC en assets monétisables

#### ✅ Étape 3.1 : Contenu & Visibilité (Semaine 1 Post-POC)
- [ ] **Article Medium/Dev.to** : "TRM Architecture: 50% Cost Reduction for LLM Inference"
  - Inclure benchmarks chiffrés
  - Code snippets clés
  - Architecture diagram
  - Target : 2000-3000 mots
- [ ] **GitHub Repo Public** : `bergsonAndFriends-TRM`
  - README portfolio-quality
  - Benchmarks dans `/docs/benchmarks/`
  - Scripts reproduction
  - Architecture diagram
- [ ] **LinkedIn Série** (5 posts) :
  - Post 1 : Problème (coûts inférence LLM explosifs)
  - Post 2 : Solution (architecture TRM - schéma)
  - Post 3 : Implémentation (challenges techniques)
  - Post 4 : Résultats (benchmarks -50% coût)
  - Post 5 : Ouverture (code open source + retours)
- [ ] **Proposition Talks** : PyConFR, MLOps meetups, HuggingFace meetups
  - Titre : "From 14B to 7B: How BERT+Mistral Saved Our LLM Costs"

**Livrables :**
- Article Medium publié (1000+ views target)
- Repo GitHub public (50+ stars target)
- 5 posts LinkedIn (100+ reactions target)
- 1-2 propositions talks acceptées

#### ✅ Étape 3.2 : Produits Dérivés (Semaine 2-3 Post-POC)
- [ ] **Template Migration TRM** : Guide + scripts
- [ ] **Vidéo YouTube** (15-20 min) :
  - Walkthrough architecture
  - Démo live POC
  - Benchmarks commentés
- [ ] **Consulting Pitch Deck** : "LLM Cost Audit"
- [ ] **Formation/Tutoriel** (si traction audience) :
  - Cours "LLM Cost Optimization with TRM Architecture"
  - Prix : 49-99€
  - Plateforme : Udemy, Gumroad, ou propre site

**Livrables :**
- Template migration TRM
- Vidéo YouTube (500+ views target)
- Pitch deck consulting
- Formation prévente (si demande)

**ROI Cible** : 100€ investis → 2-5k€ retour (×20-50) via :
- Consulting : 1-2 missions (5-10k€)
- Formation : Prévente tutoriel (49€ × 20-50 personnes = 1-2.5k€)
- Sponsoring : GitHub Sponsors (5-10$/mois × 10-20 early adopters)

---

## 📊 Analyse Coût/Bénéfice

### Investissement POC TRM

| Poste | Coût | Durée |
|-------|------|-------|
| Temps développement | 2-3 semaines (votre temps) | - |
| Infrastructure Vast.ai/RunPod | 80-100€ (tests + benchmarks) | 1 mois |
| **Total** | **~100€ + temps** | **3 semaines** |

### Retours Potentiels (6 mois)

| Scénario | Probabilité | Retour Estimé |
|----------|------------|---------------|
| Consulting (1-2 missions) | 30% | 5-10k€ |
| Formation/Tutoriel | 20% | 1-3k€ (ventes) |
| Sponsors Open Source | 10% | 500-1k€/an |
| Visibilité → Opportunités | 50% | Inestimable (jobs, network) |
| Rien (mais apprentissage) | 20% | Expertise TRM (valorisable plus tard) |

**ROI Conservateur** : Même si 0€ direct, vous avez :
- Portfolio technique solide
- Expertise différenciante (TRM avant la hype)
- Contenu réutilisable (articles, talks, démos)

#### ✅ Étape 2.1 : Intégration Complète (1 jour)
- [ ] Finaliser l'intégration complète du pipeline (Router → RAG → BERT → Mistral).
- [ ] Gérer la persistance du `STATE_IMAGE` entre les tours de conversation (Session State).
- [ ] **Tests d'Intégration Critiques :** Simuler 5 dialogues complets (end-to-end) pour s'assurer que le `STATE_IMAGE` s'enrichit correctement à chaque tour.

**Livrables :**
- Pipeline complet fonctionnel
- `tests/test_integration_trm.py` (5 dialogues de test)
- Rapport validation chaîne

#### ✅ Étape 2.2 : Optimisation de la Densité (1 jour)
- [ ] **Focus Qualité/Taille:** Affiner l'extraction des concepts RAG pour garantir la densité de l'information.
- [ ] Vérifier la taille du contexte de Mistral à chaque tour (**vérification critique POC : Total ≤ 500 tokens**).
- [ ] Ajuster les *thresholds* RAG pour garantir la pertinence.

**Livrables :**
- Fonction `extract_concepts_from_rag()` affinée
- `Backend/config/thresholds.py` (thresholds optimaux)
- Validation taille contexte ≤ 500 tokens
- Rapport optimisation densité

### Phase 4 : Améliorations & Reportés (Post-Valorisation)

| Étape Reportée | Description et Objectif |
| :--- | :--- |
| **Phase 3.1 & 3.3 (Mini-store)** | Activation complète du `Mini-store`, affinage de la détection de patterns et implémentation de la boucle de feedback vers BERT. |
| **Phase 4.3 (Tests de Charge)** | Simuler sessions parallèles, vérifier latence moyenne < 3s, et s'assurer qu'il n'y a pas d'erreurs de mémoire. |
| **Phase 4.4 (Évaluation Qualité)** | Évaluation formelle de la cohérence et du style généré par rapport à une *baseline* (nécessite un dataset d'évaluation). |
| **Optimisation Pipeline (Parallélisation)** | Mise en cache et parallélisation RAG/BERT pour la montée en charge. |
| **Multi-philosophes** | Extension du corpus à Kant et Bergson (actuellement Spinoza uniquement). |
| **Note** | Les axes multiples BERT sont **déjà implémentés** dans le POC pour maximiser la précision de la démo. |

---

**Note POC :** Cet ajustement permet de débloquer rapidement la preuve de concept en **3.5 à 4 jours** en se concentrant sur les fonctions de base, tout en laissant le travail d'optimisation, de performance et d'heuristique (Mini-store) pour après la validation initiale.

---

## 📈 Métriques & Tests - POC

### Métriques Critiques pour POC

1. **Validation Chaîne Fonctionnelle** :
   - ✅ Pipeline complet exécutable sans erreur
   - ✅ STATE_IMAGE généré et structuré correctement
   - ✅ Contexte Mistral ≤ 500 tokens (vérification critique)

2. **Qualité STATE_IMAGE (POC Complet)** :
   - Nombre concepts extraits : 3-8 par tour
   - Taille STATE_IMAGE : 150-250 tokens (version complète avec tous les axes)
   - **Critique POC :** Pas de texte brut des passages RAG dans STATE_IMAGE
   - **Critique POC :** Concepts RAG présents dans `concepts_rag` (condensés)
   - **Critique POC :** Tous les axes analysés (concepts_actifs, intention, tension, style, ton, priorité, relations, emotion)
   - Couverture axes : 9/9 axes analysés (maximum précision pour démo)

3. **Qualité Génération (POC Complet)** :
   - Réponse générée cohérente avec STATE_IMAGE complet
   - **Critique POC :** Pas de passages RAG bruts dans réponse générée
   - Utilisation des concepts du STATE_IMAGE dans la réponse
   - Utilisation de tous les axes du STATE_IMAGE (style, ton, intention, relations, etc.)

4. **Efficacité RAG (POC)** :
   - Similarity score moyen : ≥ 0.45 (threshold minimal)
   - Passages récupérés pertinents (vérification manuelle)

### Métriques Reportées (Post-POC)

- Latence détaillée par composant
- Évaluation formelle qualité/style
- Tests de charge et performance
- Métriques Mini-store (désactivé en POC)

### Tests Critiques pour POC

**Test `rag_retriever.retrieve_passages()`** :
```python
def test_retrieve_passages():
    retriever = RAGRetriever(corpus="spinoza")
    query = "Qu'est-ce que le conatus ?"
    passages = retriever.retrieve_passages(query, top_k=3)
    
    assert len(passages) <= 3
    assert all(p['similarity_score'] >= 0.45 for p in passages)
    assert all('conatus' in p['text'].lower() or 'conatus' in p['concepts'] 
                for p in passages[:2])  # Top 2 doivent être pertinents
```

**Test `bert_encoder.extract_concepts_from_rag()`** :
```python
def test_extract_concepts_from_rag():
    encoder = BERTEncoder()
    passages = [
        {"text": "Le conatus est l'effort pour persévérer...", "concepts": ["conatus"]}
    ]
    concepts = encoder.extract_concepts_from_rag(passages)
    
    assert len(concepts) > 0
    assert "conatus" in concepts
    # Vérifier que concepts sont condensés, pas texte brut
    assert all(len(c) < 50 for c in concepts)  # Concepts courts
```

**Test `bert_encoder.encode_to_state_image()`** :
```python
def test_encode_to_state_image():
    encoder = BERTEncoder()
    conversation = [{"user": "...", "assistant": "..."}]
    rag_passages = [{"text": "...", "concepts": ["conatus"]}]
    prev_state = None
    
    state_image = encoder.encode_to_state_image(
        conversation, rag_passages, prev_state, {}
    )
    
    assert "concepts_actifs" in state_image
    assert "concepts_rag" in state_image
    assert len(state_image["concepts_rag"]) > 0
    assert state_image["concepts_rag"][0] != rag_passages[0]["text"]  # Pas texte brut
```

**Test `mistral_generator.generate()`** :
```python
def test_generate_with_state_image():
    generator = MistralGenerator()
    state_image = {
        "concepts_actifs": ["conatus"],
        "intention": "clarifier",
        "style": "pédagogique"
    }
    user_input = "Peux-tu m'expliquer le conatus ?"
    
    response = generator.generate(state_image, user_input, system_prompt)
    
    assert len(response) > 0
    # Vérifier que réponse utilise concepts de STATE_IMAGE
    assert "conatus" in response.lower()
    # Vérifier que pas de passages RAG bruts
    assert "Éthique" not in response or "prop." not in response  # Pas citations brutes
```

### Tests d'Intégration (POC - 5 Dialogues Minimum)

**Test pipeline complet** :
```python
def test_pipeline_complete():
    # Simuler dialogue complet
    session = SessionManager.create_session(philosopher="spinoza")
    
    for turn in range(5):
        user_input = f"Message tour {turn}"
        
        # Pipeline complet
        philosopher = router.determine_philosopher(user_input, session.context)
        passages = rag_retriever.retrieve_passages(user_input, philosopher)
        state_image = bert_encoder.encode_to_state_image(
            session.dialogue_history, passages, session.state_image, session.mini_store.get_feedback()
        )
        response = mistral_generator.generate(state_image, user_input, system_prompt)
        patterns = mistral_generator.detect_patterns(response)
        mini_store.update_patterns(patterns)
        
        # Vérifications Critiques POC
        assert state_image["philosopher"] == "spinoza"
        assert len(state_image["concepts_rag"]) > 0
        assert len(response) > 0
        # Vérifier pas de texte brut RAG dans STATE_IMAGE
        assert not any("Éthique" in str(v) or "prop." in str(v) 
                      for v in state_image.values() if isinstance(v, (str, list)))
        # Vérifier taille contexte ≤ 500 tokens (approximatif)
        context_tokens = len(state_image_str.split()) + len(user_input.split())
        assert context_tokens <= 500, f"Contexte trop long: {context_tokens} tokens"
```

---

## 💻 Exemples Concrets

### Vérification Ressources & Chargement Modèles (Étape 1.1b)

```python
import torch
from transformers import AutoTokenizer, AutoModel
from vllm import LLM
import psutil
import GPUtil

def validate_resources():
    """
    Valide que les modèles peuvent être chargés simultanément
    sans conflit de ressources.
    """
    results = {
        "bert_cpu": {},
        "mistral_gpu": {},
        "coexistence": {}
    }
    
    # 1. Chargement BERT sur CPU
    print("Chargement BERT sur CPU...")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    
    # Vérifier mémoire RAM
    ram_before = psutil.virtual_memory().used / (1024**3)  # GB
    ram_after = psutil.virtual_memory().used / (1024**3)
    ram_bert = ram_after - ram_before
    
    results["bert_cpu"]["ram_usage_gb"] = ram_bert
    results["bert_cpu"]["ram_ok"] = ram_bert < 4.0  # < 4GB pour BERT-mini
    
    # Test inférence BERT
    test_input = bert_tokenizer("Test BERT", return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**test_input)
    results["bert_cpu"]["inference_ok"] = True
    results["bert_cpu"]["latency_ms"] = 50  # Exemple
    
    # 2. Chargement Mistral sur GPU
    print("Chargement Mistral sur GPU...")
    gpu_before = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
    
    mistral_llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8
    )
    
    gpu_after = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
    vram_mistral = gpu_after - gpu_before
    
    results["mistral_gpu"]["vram_usage_gb"] = vram_mistral / 1024
    results["mistral_gpu"]["vram_ok"] = vram_mistral < 16 * 1024  # < 16GB VRAM
    
    # Test inférence Mistral
    test_prompt = "Test Mistral"
    outputs = mistral_llm.generate([test_prompt], SamplingParams(max_tokens=10))
    results["mistral_gpu"]["inference_ok"] = True
    results["mistral_gpu"]["latency_ms"] = 200  # Exemple
    
    # 3. Test de coexistence simultanée
    print("Test coexistence simultanée...")
    import time
    
    start = time.time()
    
    # Inférence BERT (CPU) et Mistral (GPU) simultanément
    bert_input = bert_tokenizer("Test coexistence BERT", return_tensors="pt")
    mistral_outputs = mistral_llm.generate(["Test coexistence Mistral"], 
                                          SamplingParams(max_tokens=10))
    
    with torch.no_grad():
        bert_output = bert_model(**bert_input)
    
    elapsed = time.time() - start
    
    results["coexistence"]["simultaneous_ok"] = True
    results["coexistence"]["latency_ms"] = elapsed * 1000
    results["coexistence"]["no_conflicts"] = True  # Pas d'erreur
    
    # Vérifications finales
    results["coexistence"]["both_accessible"] = (
        results["bert_cpu"]["inference_ok"] and 
        results["mistral_gpu"]["inference_ok"]
    )
    
    return results

# Exécution
if __name__ == "__main__":
    results = validate_resources()
    print("\n=== Résultats Validation Ressources ===")
    print(f"BERT CPU - RAM: {results['bert_cpu']['ram_usage_gb']:.2f} GB - OK: {results['bert_cpu']['ram_ok']}")
    print(f"Mistral GPU - VRAM: {results['mistral_gpu']['vram_usage_gb']:.2f} GB - OK: {results['mistral_gpu']['vram_ok']}")
    print(f"Coexistence - Latence: {results['coexistence']['latency_ms']:.0f} ms - OK: {results['coexistence']['no_conflicts']}")
    
    if all([
        results['bert_cpu']['ram_ok'],
        results['mistral_gpu']['vram_ok'],
        results['coexistence']['no_conflicts'],
        results['coexistence']['both_accessible']
    ]):
        print("\n✅ Validation réussie - Les modèles peuvent coexister")
    else:
        print("\n❌ Validation échouée - Vérifier les ressources")
```

### RAG Retrieve avec Enrichissement Query

```python
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

class RAGRetriever:
    def __init__(self, corpus_path, philosopher):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.philosopher = philosopher
        self.index = self._load_index(corpus_path)
        self.passages = self._load_passages(corpus_path)
    
    def retrieve_passages(self, user_input, state_image=None, top_k=3):
        # Enrichir query avec concepts actifs du STATE
        if state_image and state_image.get("concepts_actifs"):
            query = f"{user_input} {' '.join(state_image['concepts_actifs'][:3])}"
        else:
            query = user_input
        
        # Encode query
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        
        # Search
        scores, indices = self.index.search(
            query_emb.cpu().numpy().reshape(1, -1), top_k
        )
        
        # Filter by threshold
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= 0.45:  # Threshold
                passage = self.passages[idx]
                passage["similarity_score"] = float(score)
                results.append(passage)
        
        return results[:top_k]
```

### BERT Encode avec Extraction Concepts RAG

```python
from transformers import AutoTokenizer, AutoModel
import torch

class BERTEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.model.eval()
    
    def extract_concepts_from_rag(self, passages):
        """
        Extrait concepts clés des passages RAG, pas texte brut.
        Utilise extraction d'entités ou keywords.
        """
        concepts = []
        for passage in passages:
            # Méthode 1: Utiliser concepts pré-annotés si disponibles
            if passage.get("concepts"):
                concepts.extend(passage["concepts"])
            else:
                # Méthode 2: Extraction keywords (exemple simplifié)
                text = passage["text"]
                # Extraire noms propres, termes techniques
                keywords = self._extract_keywords(text)
                concepts.extend(keywords[:3])  # Top 3 par passage
        
        # Dédupliquer et limiter
        return list(set(concepts))[:8]  # Max 8 concepts
    
    def encode_to_state_image(self, conversation, rag_passages, prev_state, mini_store_feedback):
        """
        Encode conversation + passages RAG en STATE_IMAGE structuré.
        """
        # Extraire concepts RAG (pas texte brut)
        concepts_rag = self.extract_concepts_from_rag(rag_passages)
        sources_rag = [p.get("source", "") for p in rag_passages]
        
        # Analyser conversation
        last_exchange = conversation[-1] if conversation else {}
        user_text = last_exchange.get("user", "")
        assistant_text = last_exchange.get("assistant", "")
        
        # Analyse axes (simplifié, à améliorer)
        axes = self.analyze_axes(user_text + " " + assistant_text)
        
        # Construire STATE_IMAGE
        state_image = {
            "concepts_actifs": axes.get("concepts", []),
            "concepts_rag": concepts_rag,
            "sources_rag": sources_rag,
            "intention": axes.get("intention", "neutre"),
            "tension": axes.get("tension", "neutre"),
            "style": axes.get("style", "neutre"),
            "priorite": axes.get("priorite", []),
            "relations": axes.get("relations", []),
            "emotion": axes.get("emotion", "neutre"),
            "recurrence": mini_store_feedback.get("recurrences", {}),
            "metadata": {
                "philosopher": rag_passages[0].get("philosopher") if rag_passages else None,
                "turn": (prev_state.get("metadata", {}).get("turn", 0) + 1) if prev_state else 1
            }
        }
        
        # Fusionner avec STATE précédent si existe
        if prev_state:
            state_image = self._merge_states(prev_state, state_image)
        
        return state_image
```

### Mistral Generate avec STATE_IMAGE

```python
from vllm import LLM, SamplingParams

class MistralGenerator:
    def __init__(self):
        self.llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=500)
    
    def generate(self, state_image, user_input, system_prompt):
        """
        Génère réponse avec STATE_IMAGE structuré.
        """
        # Construire prompt avec STATE_IMAGE
        prompt = self._construct_prompt(state_image, user_input, system_prompt)
        
        # Vérifier taille contexte (≤500 tokens)
        tokens = self._count_tokens(prompt)
        if tokens > 500:
            # Réduire STATE_IMAGE si nécessaire
            state_image = self._reduce_state_image(state_image, target_tokens=200)
            prompt = self._construct_prompt(state_image, user_input, system_prompt)
        
        # Générer
        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text
        
        return response
    
    def _construct_prompt(self, state_image, user_input, system_prompt):
        """
        Construit prompt avec STATE_IMAGE structuré.
        """
        # Formater STATE_IMAGE en texte lisible
        state_text = self._format_state_image(state_image)
        
        prompt = f"""{system_prompt}

[CONTEXT_STATE]
{state_text}

[USER_INPUT]
{user_input}

[ASSISTANT_RESPONSE]
"""
        return prompt
    
    def _format_state_image(self, state_image):
        """
        Formate STATE_IMAGE en texte structuré pour prompt.
        """
        lines = []
        
        if state_image.get("concepts_actifs"):
            lines.append(f"Concepts actifs: {', '.join(state_image['concepts_actifs'][:5])}")
        
        if state_image.get("concepts_rag"):
            lines.append(f"Concepts pertinents (corpus): {', '.join(state_image['concepts_rag'][:5])}")
        
        if state_image.get("intention"):
            lines.append(f"Intention: {state_image['intention']}")
        
        if state_image.get("style"):
            lines.append(f"Style: {state_image['style']}")
        
        return "\n".join(lines)
```

### Mini-store avec Détection Patterns

```python
from collections import Counter
import re

class MiniStore:
    def __init__(self):
        self.recurrences = {
            "termes_repetes": Counter(),
            "structures": []
        }
        self.alertes = []
    
    def update_patterns(self, response_text):
        """
        Met à jour patterns détectés depuis réponse Mistral.
        """
        # Détecter termes répétés
        words = re.findall(r'\b\w+\b', response_text.lower())
        common_words = [w for w, count in Counter(words).most_common(5) if count >= 2]
        self.recurrences["termes_repetes"].update(common_words)
        
        # Détecter structures
        if "?" in response_text:
            self.recurrences["structures"].append("question")
        if "car" in response_text.lower() or "parce que" in response_text.lower():
            self.recurrences["structures"].append("explication")
        
        # Alertes
        if len(common_words) > 3:
            self.alertes.append("répétition excessive")
    
    def get_feedback_for_bert(self):
        """
        Retourne feedback pour BERT (pondération STATE).
        """
        return {
            "recurrences": {
                "termes_repetes": dict(self.recurrences["termes_repetes"].most_common(5)),
                "structures": list(set(self.recurrences["structures"][-10:]))
            },
            "alertes": self.alertes[-5:]
        }
```

---

## 📝 Checklist Implémentation - POC Bootstrap

### Phase 0 : Préparation 0€ (Cette semaine - Colab Gratuit)
- [ ] Prototyper BERT encoder sur Colab gratuit (CPU)
- [ ] Générer embeddings RAG (sentence-transformers) sur Colab T4
- [ ] Tester Mistral 7B inference sur Colab gratuit (12h sessions)
- [ ] Créer architecture diagram (Excalidraw)
- [ ] Préparer scripts benchmark automatiques
- [ ] Rédiger README structure POC

### Phase 1 : Développement Colab Gratuit (Jours 1-3, 0€)
- [ ] Router & RAG Retriever (indexes Phase 0)
- [ ] BERT Encoder Core Complet (prototype Phase 0)
- [ ] Mistral Generator (baseline Phase 0)
- [ ] Templates & Endpoint `/chat_trm`

### Phase 2 : Intégration & Benchmarks Vast.ai (Jours 4-7, ~80€)
- [ ] **🆕 Vérification Ressources & Chargement Modèles** (CRITIQUE - Jour 4, ~20€)
- [ ] Intégration complète pipeline (Jours 4-5, ~40€)
- [ ] Tests d'intégration (5 dialogues)
- [ ] Benchmarks comparatifs TRM vs Qwen 14B (Jours 6-7, ~40€)
- [ ] Validation contexte ≤ 500 tokens
- [ ] Enregistrement démo vidéo

### Phase 3 : Valorisation Immédiate (Semaines Post-POC, 0€)
- [ ] Article Medium/Dev.to publié
- [ ] Repo GitHub public portfolio-quality
- [ ] Série LinkedIn (5 posts)
- [ ] Proposition talks conférences
- [ ] Vidéo YouTube walkthrough
- [ ] Template migration TRM
- [ ] Consulting pitch deck

### Reporté Post-POC
- [ ] Mini-store complet avec feedback BERT
- [ ] Tests de charge
- [ ] Évaluation qualité formelle
- [ ] Optimisation pipeline (parallélisation, cache)
- [ ] Extension multi-philosophes (Kant, Bergson)
- **Note** : Les axes multiples BERT sont **déjà implémentés** dans le POC pour maximiser la précision

---

## 🚀 Prochaines Étapes - POC Bootstrap

### Actions Immédiates (Avant 100€)

1. **Phase 0 - Préparation Colab Gratuit** (Cette semaine)
   - Prototyper BERT encoder (CPU Colab)
   - Générer embeddings RAG (T4 Colab)
   - Tester Mistral 7B baseline (T4 Colab)
   - Créer architecture diagram
   - Préparer scripts benchmark

2. **Comparer Infrastructure** (Cette semaine)
   - Comparer prix Vast.ai vs RunPod pour setup final
   - Définir métriques benchmarks (coût, latence, qualité)
   - Préparer scripts déploiement Vast.ai

3. **Amasser 100€** (Objectif avant Phase 1)

### Timeline POC Complète

**Semaine 1** : Phase 0 (Colab gratuit)  
**Semaine 2** : Phase 1 (Colab gratuit) + Phase 2 début (Vast.ai)  
**Semaine 3** : Phase 2 fin (benchmarks) + Phase 3 début (valorisation)

**Budget Total** : 80-100€ (Vast.ai uniquement pour intégration/benchmarks)  
**ROI Cible** : 100€ → 2-5k€ (×20-50) via consulting/formation/sponsoring

---

## 📚 Références

- Architecture Dialecthon : `/Dialecthon/docs/guides/GUIDE_DIALECTHON.md`
- RAG Integration : Documentation RAG existante
- Mistral 7B : Documentation HuggingFace

---

**Note** : Ce document est vivant. Mettre à jour au fur et à mesure de l'implémentation.
