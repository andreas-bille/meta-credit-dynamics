# Strukturierte AI-Literaturliste & Technologie-Zusammenfassung

## Geschätzte Studienzeitdauer
Die Dauer hängt stark vom persönlichen Tempo, Vorkenntnissen (als Theoretischer Physiker habe ich die mathematischen Kenntnisse) und dem Grad des praktischen Arbeitens ab. Technische Fachbücher erfordern oft eine intensive Auseinandersetzung, Code-Implementierung und tiefergehendes Verständnis, was die Lesezeit pro Seite im Vergleich zu Romanen erheblich verlängert (durchschnittlich 5-6 Minuten pro Seite für technisches Material).

**Annahmen**

- 15 Bücher in der Liste.
- Durchschnittlich 400 Seiten pro Buch (geschätzt).
- Intensives Studium und Coden (entspricht etwa 1-2 Stunden pro 20 Seiten).

| Szenario |	Stunden pro Woche	| Geschätzte Gesamtdauer
| --- | --- | ---
| **Hobby (Geringe Intensität)** |	5 - 10 Stunden |	12 - 24 Monate
| **Fokus (Mittlere Intensität)** | 	15 - 20 Stunden |	6 - 8 Monate
| **Vollzeit-Studium** |	35+ Stunden	| 3 - 4 Monate


## Konsolidierte Leseliste & Technologie-Schwerpunkte
| Phase	| Fokus	Bücher | (Manning MEAP/Bücher)
| --- | --- | --- |
| **Phase 0** |Mathematik & Theorie	|Math_and_Architectures_of_Deep_Learning 
| **Phase 1**	|Frameworks & Grundlagen	|Deep_Learning_with_PyTorch_Second_Editi_v14_MEAP, Hugging_Face_in_Action
| **Phase 2**	|Architekturen & SLMs	|Build_a_Large_Language_Model_(From_Scratch), Domain-Specific_Small_Language_Models_v8_MEAP
| **Phase 3**	|Erweitertes Reasoning & Kausalität (mit Graphen & RAG)	|Build_a_Reasoning_Model_(From_Scratch)_v3_MEAP, Build_a_DeepSeek_Model_(From_Scratch)_v2_MEAP, Knowledge_Graphs_and_LLMs_in_Action, Essential_GraphRAG, Graph_Neural_Networks_in_Action, Causal_AI
| **Phase 4**	|Autonome Agenten & RL	|AI_Agents_in_Action, AI_Agents_and_Applications_v8_MEAP, The_RLHF_Book_v1_MEAP

## Kurzzusammenfassung der Kerntechnologien
### Retrieval-Augmented Generation (RAG) & GraphRAG
- **Definition**: Eine Technik, die einem Sprachmodell erlaubt, auf externe, aktuelle Wissensdatenbanken zuzugreifen, bevor es eine Antwort generiert.
- **Zweck**: Reduziert "Halluzinationen" (erfundene Fakten) des Modells und ermöglicht die Nutzung von proprietären, nicht-öffentlichen Daten (z.B. Unternehmensdokumente).
- **GraphRAG**: Eine spezielle Form von RAG, bei der die externen Daten als strukturierter Wissensgraph gespeichert werden, was komplexeres und präziseres Reasoning ermöglicht.

### Graph Neural Networks (GNNs)
- **Definition**: Eine Klasse von Deep-Learning-Modellen, die Graphen als Eingabestruktur verwenden und Informationen über die Verbindungen (Kanten) und Knoten aggregieren.
- **Zweck**: Ideal zur Analyse von vernetzten Daten (soziale Netzwerke, Moleküle, Wissensgraphen) und zur Vorhersage von Beziehungen oder Eigenschaften innerhalb des Graphen.

### AI Agents (KI-Agenten)
- **Definition**: Autonome Softwaresysteme, die ein zugrunde liegendes LLM/SLM als "Gehirn" nutzen, um Aufgaben zu planen, Tools zu verwenden (z.B. API-Aufrufe) und in einer Umgebung Aktionen durchzuführen.
- **Zweck**: Ermöglicht komplexe, mehrstufige Zielerreichung ohne menschliche Interaktion und führt zu dynamischeren, intelligenteren Anwendungen als einfache Chatbots.

### Reinforcement Learning (RL)
- **Definition**: Eine Art des maschinellen Lernens, bei dem ein Agent durch "Versuch und Irrtum" in einer Umgebung lernt, um eine Belohnungsfunktion zu maximieren.
- **Zweck bei AI**: Wird hauptsächlich als RLHF (Reinforcement Learning from Human Feedback) genutzt, um Sprachmodelle an menschliche Präferenzen anzupassen und ethischer/hilfreicher zu machen. Bei Agenten ermöglicht es die autonome strategische Entscheidungsfindung und Tool-Nutzung.