# Phase D – Reparatur und Stabilisierung

**Status:** Konzept v0.3 – implementierbar  
**Kontext:** v0.2.0 → Phase D  
**Prinzipien:** semantikfrei · skaleninvariant · nicht-optimierend

---

## Ziel

Phase D dient nicht der Rettung oder Optimierung des Systems, sondern der
**forensischen Analyse minimaler Reparaturmechanismen**.

Untersucht wird:
- was kaputtgeht
- warum es kaputtgeht
- welche neuen Brüche durch Reparatur entstehen

> **Leitmotiv:**  
> *Reparieren heißt hier: Zeit kaufen, um beim Sterben zuzusehen.*

---

## 1. Ausgangslage

Phase C (v0.2.0) ist abgeschlossen.  
Das System weist inhärente Instabilitäten auf, die durch Reparaturmechanismen
temporär abgefedert werden können.

Phase D:
- friert Architektur und Lernregeln ein
- erlaubt ausschließlich **minimale, reversible Eingriffe**

**Nicht-Ziele:**
- Systemrettung
- Effizienzsteigerung
- Fairness
- langfristige Stabilisierung

---

## 2. Grundannahme

Reparaturmechanismen erzeugen keine Sicherheit, sondern **Zeit**.  
Zeit ist kein neutraler Speicher, sondern akkumuliert Nebenwirkungen.

> **Reparatur = Meta-Kredit**  
> bewusste Verschiebung von Risiko in die Zukunft ohne Tilgung.

**Ergänzung:**  
Zeitverschiebung ist **puffergebunden**.  
Ohne gespeicherten Puffer (Wealth/Reserve) ist Entscheidungsaufschub nicht möglich.

**Warnhinweis:**  
Aus „Puffer“ darf keine normative Verteilung („wer Zeit verdient“) abgeleitet werden.

---

## 3. Meta-Kredit (Oberkategorie)

**Definition:**  
Meta-Kredit ist die Fähigkeit eines Systems, negative Zustände in zukünftige
Zustände umzubuchen, ohne sie aktuell aufzulösen.

**Eigenschaften:**
- kein expliziter Schuldner
- kein Rückzahlungstermin
- keine klare Haftung

Meta-Kredit ist **keine Institution**, sondern eine **Systemfunktion**.

---

## 4. Markt ohne Marktlevel

Meta-Kredit erzeugt keinen klassischen Markt, sondern einen impliziten
**Zeitaufschub- und Risikodiffusionsmarkt**.

### Kein Marktlevel

Es gibt **keine Markt-Ebene** und keinen Marktindex als Levelmarker, weil:

- **Skalenvarianz:** gleiche Dynamik auf allen Skalen
- **Semantikfreiheit:** Einheiten kennen kein „Level“
- **Praktisch:** lokale Signale müssen systemweit lesbar sein

> **Markt ist ein Zustand offener Kopplung, kein Ort.**

**Warnhinweis:**  
Ein Marktlevel erzeugt zwangsläufig Hierarchie, Gatekeeping und Intransparenz.

---

## 5. Emergenz von Brokerpfaden (ohne Akteure)

Meta-Kredit selektiert implizite **Brokerpfade** – Pfade, entlang derer Risiko
geparkt wird.

**Charakteristika:**
- gehandelt wird Zeitaufschub
- Preise sind implizit (Tail-Risiken, Zentralität)
- Broker werden selektiert, nicht gewählt

> **Nicht Broker wählen Stacks – Stacks wählen Broker.**

Broker sind **Pfadfunktionen**, keine Akteure.

**Warnhinweis:**  
Keine Broker-Agenten, keine Broker-Typen, keine Brokermärkte bauen.

---

## 6. Broker als Inhibitoren an der Grenze

Emergente Broker wirken inhibierend, sind aber keine altruistischen Stabilisatoren.

**Selektionsdruck:**
- zu wenig Absorption → Irrelevanz
- zu viel Absorption → Systemrelevanz → Überlastung

> **Inhibitoren überleben nur an der Grenze.**

**Ergänzung:**  
Systemweit lesbare Stresssignale können **Fressattacken / Thrashing** auslösen.
Dies ist eine Sterbe-Signatur, kein Bug.

---

## 7. Scheidung und Rückabwicklung

### 7.1 Scheidung

**Scheidung** = Auflösung einer Kopplung ohne Tilgung der akkumulierten Schuld.

**Risiken:**
- Re-Exposition von Varianz
- Verlust von Verzögerungseffekten
- Scheidungskosten

> **Scheidung ist ein regulärer Ausgang – aber nie kostenlos.**

Phase D **erlaubt Scheidung**, sie erzwingt sie nicht.

---

### 7.2 Rückabwicklung (Beobachtungsbegriff)

**Rückabwicklung** bezeichnet das graduelle Entlassen einer Kopplung entlang ihrer
Entstehungslogik.

In Phase D:
- kein eigener Mechanismus
- reine Interpretations- und Beobachtungskategorie

**Warnhinweis:**  
Keine Rückabwicklungs-Engine bauen (führt zu Institutionalisierung).

---

## 8. Erlaubte Reparaturmechanismen

Zulässig nur, wenn sie:
1. keine neue Information einführen
2. keine globale Optimierung betreiben
3. nur Eskalation verzögern

**Klassen:**
- Flussbegrenzung (Caps)
- Lag Injection
- Soft Bail-out
- Lokale Isolation

Alle Mechanismen:
- reversibel
- abschaltbar
- telemetrisch markiert

---

## 9. Nicht erlaubte Maßnahmen

- Re-Training
- neue Reward-Funktionen
- adaptive Optimierer
- Smart Repairs
- Fairness- oder Effizienzkorrekturen

> *Alles, was sich vernünftig anfühlt, ist verboten.*

---

## 10. Sterbe-Telemetrie

Phase D bewertet **Sterbesignaturen**, nicht Performance:

- Zeit bis Kollaps
- Kollapskaskaden (Tiefe/Breite)
- Reparaturabhängigkeit
- Varianz nach Stabilisierung
- Emergenz unsterblicher Knoten

---

## 11. Fehlende Nachfrage als Exploitations-Signatur

Exploitation ist häufig stabil, weil **Exit-Optionen fehlen**.

Phase D misst daher zusätzlich:
- Anzahl alternativer Kopplungsangebote
- Zeit seit letztem Angebot
- Angebotsrate

> **Ein leeres Orderbuch ist selbst Information.**

**Wichtig:**  
Diese Größen sind **reine Telemetrie**, keine Entscheidungsregeln.

**Warnhinweis:**  
Stack-Index oder Integrationshöhe dürfen nicht deterministisch
Exit-Chancen verbieten.

---

## 12. Zentrale Erkenntnis

Versicherungen, Bail-Outs und Derivate sind **Symptome**, keine Ursachen.

> Wenn man über Bail-Outs reden muss, ist das Modell noch nicht fertig.

Phase D zeigt außerdem:
> Exploitation bleibt stabil, solange sie systemisch günstiger ist als ihr Ende.

---

## 13. Abschluss Phase D

Phase D ist abgeschlossen, wenn:

- Reparatur Fragilität erhöht
- Brokerpfade systemrelevant werden
- Stabilität Wahrheit verdrängt
- Scheidung als regulärer Ausgang beobachtbar ist
- fehlende Nachfrage sichtbar wird

> **Kanonischer Satz:**  
> Reparatur erzeugt Meta-Kredit.  
> Meta-Kredit selektiert Brokerpfade.  
> Broker überleben an der Grenze.  
> Scheidung ist möglich – aber nie kostenlos.  
> Exploitation bleibt stabil, wenn Exit fehlt.

---

## 14. Übergang zu Phase E

Phase E beginnt, sobald:
- Telemetrie rückgekoppelt wird
- Broker zu Akteuren werden
- Governance, Normen oder Optimierung eingeführt werden
