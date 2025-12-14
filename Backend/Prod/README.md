# TRM Backend Production

Code de production pour le système TRM.

## Structure Attendue

```
Prod/
├── api/              # FastAPI endpoints
│   └── endpoints/
│       └── trm.py
├── models/           # Gestion modèles IA
│   ├── router.py
│   ├── rag_retriever.py
│   ├── bert_encoder.py
│   ├── mistral_generator.py
│   └── mini_store.py
├── config/           # Configuration
│   └── thresholds.py
├── prompts/          # Prompts versionnés
│   ├── system_prompt.py
│   └── chat_template.py
└── session_manager.py
```

Voir `docs/guides/PRD_TRM_ENRICHI.md` pour l'architecture complète.

