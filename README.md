# fdi-dasi-jackson

## Equipo

* Zhaoyang Qi
* Jingyuan Wang

---

## Qué hace este proyecto

Un agente autónomo de intercambio de recursos que compite con otros agentes en un
juego multi-agente de trueque, coordinado a través de un servidor central al que
en este código llamamos **Butler**. "Butler" es el **rol** del coordinador
central definido por el protocolo del curso (registro, inventario/objetivo,
lista de pares y entregas de paquetes); la instancia concreta la provee el
profesor en la URL que indique. El agente se registra, carga su inventario y
su objetivo de victoria desde ese servidor y, a partir de ahí, mantiene una
negociación continua con sus pares: difunde sus necesidades, evalúa las
ofertas entrantes, contraoferta cuando los términos no son aceptables y
encadena intercambios intermedios para alcanzar el objetivo más rápido.

**Arquitectura en una frase:** el LLM (Ollama, mediante herramientas de
function-calling) sugiere la clasificación de intención y la estrategia; el
código aplica todas las restricciones y ejecuta todas las transiciones de
estado.

---

## Requisitos previos

- [`uv`](https://github.com/astral-sh/uv) — gestor de paquetes de Python
- [`ollama`](https://ollama.com) — runtime local de LLM
- Un modelo de Ollama con soporte de **tool-calling** (usamos `llama3.1` por
  defecto; `qwen2.5:7b-instruct` es una alternativa más rápida)

---

## Instalación

```bash
# 1. Instalar dependencias de Python
uv sync --extra dev

# 2. Descargar el modelo del LLM (una sola vez)
ollama pull llama3.1
```

---

## Ejecución

El servidor central lo provee el profesor. Antes de arrancar, ajusta
`SERVER_URL` a la URL que indique (ver tabla de configuración más abajo) y
lanza el agente:

```bash
uv run main.py
```

En el entorno de evaluación, cada agente corre en su propia máquina, por lo
que todos pueden usar el puerto por defecto (`7720`) sin colisiones.

---

## Configuración

Todos los valores se pueden sobrescribir mediante variables de entorno. Las
más útiles:

| Variable            | Valor por defecto             | Descripción                                                                       |
|---------------------|-------------------------------|-----------------------------------------------------------------------------------|
| `SERVER_URL`        | definido en `config.py`       | URL del servidor central indicada por el profesor                                 |
| `AGENT_NAME`        | definido en `config.py`       | Alias con el que se registra en el servidor central                               |
| `MY_PORT`           | `7720`                        | Puerto en el que este agente escucha `/buzon`                                     |
| `OLLAMA_MODEL`      | `llama3.1`                    | Nombre del modelo de Ollama (debe admitir tool-calling)                           |
| `OLLAMA_URL`        | `http://localhost:11434`      | URL del servidor de Ollama                                                        |
| `OLLAMA_TIMEOUT`    | `30.0`                        | Segundos antes de abandonar una llamada a Ollama                                  |
| `HTTP_TIMEOUT`      | `60.0`                        | Segundos para las llamadas HTTP entre agentes                                     |
| `BUTLER_TIMEOUT`    | `5.0`                         | Segundos para las llamadas de arranque al servidor central                        |
| `PENDING_OFFER_TTL` | `300.0`                       | Segundos antes de auto-cancelar una oferta sin respuesta                          |
| `CHAIN_RESERVE_TTL` | `180.0`                       | Segundos que se mantiene un plan de cadena sin completar                          |

---

## Protocolo con el servidor central

El agente sólo invoca **4 endpoints** sobre `SERVER_URL`. Cualquier servidor
que implemente este contrato (el del profesor o un Butler local) sirve como
coordinador:

| Método | Ruta                  | Uso                                                                 |
|--------|-----------------------|---------------------------------------------------------------------|
| `POST` | `/alias/{AGENT_NAME}` | Registrar al agente con el alias indicado                           |
| `GET`  | `/info`               | Obtener el inventario y el objetivo asignados a este agente         |
| `GET`  | `/gente`              | Listar pares activos (`[{alias, ip}, ...]`)                         |
| `POST` | `/paquete/{alias}`    | Entregar un paquete de recursos al par `alias`; cuerpo: `{recurso: cantidad}` |

Si la URL del profesor responde a estas 4 rutas con el formato esperado, no
hace falta ningún cambio en el código del agente — basta con ajustar
`SERVER_URL`.

---

## Estructura de módulos

```
main.py                Aplicación FastAPI, lifespan, endpoint /buzon, rutas del dashboard
agents.py              Cliente HTTP agente-a-agente (broadcast concurrente)
butler.py              Cliente HTTP de Butler (registro, info, lista de pares, entregas)
config.py              Configuración centralizada vía variables de entorno
decision_engine.py     Lógica de negocio central: process_request, counter, accept, chain
events.py              Pub/sub en memoria para el /api/stream del dashboard
message_normalizer.py  JSON / lenguaje natural → NormalizedMessage (regex rápida + Ollama)
messaging.py           Constructores de mensajes estructurados salientes
models.py              Esquemas Pydantic (NormalizedMessage, ChainPlan, etc.)
ollama_client.py       Cliente asíncrono del chat de Ollama con function-calling
prompt_builder.py      Todos los prompts y esquemas de herramientas de Ollama
state_manager.py       Estado seguro para async: inventario, objetivos, pendings, planes de cadena
utils.py               Helpers compartidos (whitelist de recursos, parseo de JSON)
dashboard.html         SPA embebida servida en /dashboard
```

---

## Pipeline de decisión

```
HTTP POST /buzon
   │
   │   (1) ack inmediato — devuelve {"status": "queued"} en <50 ms
   ▼
asyncio.create_task(_handle_inbox)
   │
   ▼ (en segundo plano)
message_normalizer.normalize()
   ├─ Atajo de parseo JSON
   ├─ Camino rápido por regex:
   │     "Necesito X. Puedo ofrecer Y."        → request
   │     "Te ofrezco X a cambio de Y."          → counter_offer
   │     "Te envío X."                          → delivery
   │     "¿Puedes enviarme X, como acordamos?"  → request (reclamación de honor)
   │     "ok" / "vale" / "sí" …                 → accept / clarification
   └─ Fallback: herramienta `classify_intent` de Ollama
   ▼
NormalizedMessage { kind, resources, offered_resources, ... }
   │
   ├── request        → decision_engine.process_request()
   │     0. Camino rápido honour-pending: si el par sólo está reclamando
   │        lo que ya prometimos, se liquida ahora y se omite el LLM
   │     1. Snapshot del estado (inventory, goal_needs, surplus, chain_opportunities)
   │     2. Separar prohibidos vs intercambiables (protección de recursos objetivo,
   │        teniendo en cuenta reservas intermedias)
   │     3. Las ramas de rechazo temprano INTENTAN PRIMERO UNA CONTRAOFERTA (ver abajo)
   │     4. Si no, se ejecuta la herramienta `evaluate_trade` de Ollama
   │     5. Se valida cada campo de la salida del LLM; fallback a reglas si falla
   │     6. Se ejecuta la decisión atómicamente; se dispara la segunda
   │        pierna de la cadena si aplica
   │
   ├── counter_offer → process_counter_offer()  ejecuta el flujo de request y,
   │                   si procede, genera una recontraoferta más ajustada con
   │                   la herramienta de counter de Ollama
   ├── accept    → process_accept()      honra el pending, entrega, solicita el objetivo de vuelta
   ├── delivery  → process_delivery()    suma al inventario, actualiza goal_needs
   ├── clarify   → process_clarification() envía una pregunta en español
   └── reject    → libera la reserva de cadena, limpia el pending, marca la respuesta
```

---

## Contraoferta antes del rechazo

Una solicitud de un par nunca termina en un rechazo silencioso sin antes
intentar una contraoferta. `_counter_request_if_valuable` elige una de tres
estrategias:

| Estrategia        | Cuándo                                                                       | Forma                                                                |
|-------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------|
| **concrete**      | La oferta del par ya incluye un recurso del objetivo                         | Un único recurso de surplus, cantidad pequeña, tope 3                |
| **chain**         | La oferta del par es un intermedio que otro par conocido quiere a cambio de un recurso objetivo | Un único item de surplus × un único intermedio, tope 3 |
| **speculative**   | La oferta del par no contiene nada útil                                      | Un único surplus × un único goal_need, tope 3                        |

Si la misma forma de contraoferta se repitiera contra el mismo par, el guardia
contra bucles de rechazo "endulza" el siguiente intento: primero baja en 1 el
mayor `want`; si no, sube en 1 el menor `give`. Cuando ambas direcciones se
agotan, se descarta la contraoferta y la conversación termina realmente en
reject.

---

## Intercambios en cadena (chain trades)

Cuando un par ofrece un recurso que no es del objetivo pero un tercero lo
aceptaría a cambio de un recurso objetivo, el agente automáticamente:

1. **Reserva** el intermedio entrante con `state.reserve_intermediate(plan)`
   para que otros pares no puedan llevárselo del surplus libre.
2. **Dispara** de inmediato una contraoferta de segunda pierna al tercero.
3. **Libera** la reserva cuando ocurra una de estas: el deduct completa el
   plan, el segundo tramo es rechazado, el envío del segundo tramo falla, o
   transcurre `CHAIN_RESERVE_TTL`.

La reserva se resta de `surplus` y del tope de inventario de
`_split_exchangeable`, de modo que no se puede negociar accidentalmente en
otra vía.

---

## Broadcast proactivo

Cada 45 s el agente difunde sus necesidades y su surplus actuales:

```
Necesito 3 queso, 5 aceite. Puedo ofrecer 1 queso, 3 tela, 3 aceite, 16 oro.
```

Los pares con una oferta pendiente "fresca" (más joven que
`PENDING_OFFER_TTL / 2`) se saltan en ese ciclo, para no spamear al mismo par
con la misma forma.

Los broadcasts salen **concurrentemente** (`asyncio.gather`) para que un par
inalcanzable no bloquee a los demás. La contabilidad de pendings se registra
sólo para los pares cuya entrega realmente tuvo éxito.

---

## Dashboard

Abre `http://localhost:7720/dashboard` después de iniciar el agente. SPA con:

- **Paneles de inventario y objetivo** — contadores en vivo con barras de progreso
- **Ofertas pendientes** — qué debemos en este momento y a quién
- **Modal de conversación por par** — historial completo turno a turno
- **Feed de eventos (SSE)** — flujo en tiempo real de eventos inbox /
  decision / delivery / pending_expired / chain_expired / error
- **Caja de envío manual** — enviar un mensaje puntual a cualquier par

---

## Endpoints de la API

| Método | Ruta                | Propósito                                                      |
|--------|---------------------|----------------------------------------------------------------|
| `POST` | `/buzon`            | Recibe un mensaje (sólo ack — procesamiento asíncrono)         |
| `GET`  | `/dashboard`        | SPA del dashboard                                              |
| `GET`  | `/api/state`        | Snapshot de inventario, objetivos, surplus y progreso          |
| `GET`  | `/api/agents`       | Lista de pares activos (proxy desde Butler)                    |
| `GET`  | `/api/events`       | Eventos recientes del dashboard (buffer)                       |
| `GET`  | `/api/stream`       | Flujo Server-Sent Events de eventos en vivo                    |
| `GET`  | `/api/pending`      | Todas las ofertas pendientes                                   |
| `GET`  | `/api/conversation` | Historial completo por par (`?peer=<ip>` para filtrar)         |
| `POST` | `/api/send`         | Enviar manualmente un mensaje a un par                         |
| `GET`  | `/state`            | Igual que `/api/state` (alias heredado)                        |

### Carga útil de `/buzon`

```json
{ "msg": "<cadena JSON o texto en español en lenguaje natural>" }
```

El agente responde con `{"status": "queued"}` de inmediato y procesa en
segundo plano. Todas las decisiones, contraofertas y entregas de recursos
vuelven a salir por canales separados (llamadas peer-to-peer a `/buzon` +
entregas `paquete` de Butler).

### JSON de entrada reconocido

```json
{
  "kind": "request" | "delivery" | "accept" | "reject" | "counter_offer",
  "resources": { "arroz": 2, "madera": 1 },
  "offered_resources": { "vino": 1 },
  "from_agent": "FCxxx"
}
```

### Formatos de lenguaje natural reconocidos (regex de camino rápido)

| Formato                                                  | Se interpreta como                       |
|----------------------------------------------------------|------------------------------------------|
| `Necesito 3 queso, 5 aceite. Puedo ofrecer 4 tela.`      | `request`                                |
| `Te ofrezco 3 arroz a cambio de 2 queso.`                | `counter_offer`                          |
| `Puedo ofrecer 3 arroz a cambio de 2 queso.`             | `counter_offer`                          |
| `Te doy 3 arroz a cambio de 2 queso.`                    | `counter_offer`                          |
| `Te envío 2 piedras ahora.` / `Te mando 2 piedras.`      | `delivery`                               |
| `¿Puedes enviarme 2 piedra, como acordamos?`             | `request` (reclamación de honor)         |
| `ok` / `vale` / `sí` / `acepto` / `trato` …              | `accept` si hay pending, si no `clarification` |

El formato de **reclamación de honor** es exactamente la frase que el propio
agente emite para cobrar un trato ya acordado
(`decision_engine._generate_trade_message`, rama want-only). Parsearlo de
forma determinista aquí — en vez de depender del LLM — garantiza que la
segunda pierna del trueque entre por el camino rápido honour-pending de
`process_request` y que el trato cierre limpiamente. Una afirmación escueta
("ok") sólo se considera `accept` cuando hay efectivamente una oferta
pendiente hacia ese par; en caso contrario se trata como `clarification`.

Cualquier otra cosa cae en la herramienta `classify_intent` de Ollama. Si
Ollama no está accesible o devuelve `unknown`, el mensaje se reclasifica
conservadoramente como `clarification`, de modo que se pida una aclaración
al par en lugar de ignorarlo silenciosamente.

---

## Invariantes de estado garantizadas por código (no por el LLM)

- Los **recursos reservados al objetivo** nunca se intercambian por debajo de
  la cantidad aún necesaria.
- La **whitelist de recursos** se aplica en los esquemas de las herramientas
  mediante JSON Schema `propertyNames: { enum: [...] }`, de modo que el LLM
  no puede inventar recursos.
- Las **reservas de cadena** se restan de `surplus` y de los topes de
  `exchangeable` — no hay doble gasto.
- El **inventario** nunca queda negativo (deduct atómico bajo `asyncio.Lock`).
- Las **ofertas pendientes** que fallan al enviarse por HTTP no se registran;
  el guardia contra bucles de rechazo sólo ve formas que realmente salieron.
- Las **respuestas de rechazo de los pares** ponen el `peer_response`
  correspondiente en `"rejected"` y liberan las reservas de cadena ligadas a
  esa oferta pendiente.

---

## Resolución de problemas

### `ConnectTimeout` hacia la IP de un par

La máquina del par está descartando paquetes (firewall) o esa IP está
obsoleta en Butler:

```bash
ping <ip>           # alcance en L3
nc -zv <ip> 7720    # alcance en L4
```

- `ping` responde, `nc` agota tiempo → firewall en el par (Windows Defender, UFW)
- Fallan ambos → subred incorrecta / NAT de VMware / aislamiento del AP
- `nc` devuelve `Connection refused` → el agente del par no está corriendo

### Avisos `Ollama returned no tool call`

El modelo a veces no emite una llamada de herramienta. La regex de camino
rápido ahora cubre ~80 % de los mensajes estándar, así que esto es más raro;
para el resto, el agente cae en una solicitud de clarificación. Cambiar a
`qwen2.5:7b-instruct` da un tool-calling más consistente que `llama3.1`.

### `ReadTimeout` en envíos salientes

El `/buzon` del par tarda en responder. Debería ser poco frecuente ahora que
`/buzon` hace ack en <50 ms; si aún lo ves, el par está ejecutando una
versión síncrona antigua de `/buzon`. Deberían actualizarse al código más
reciente.

### La misma IP de un par sigue apareciendo en Butler después de reiniciar

Butler persiste los registros en `estado_butler.json`. Detén Butler, borra
el archivo y reinícialo — los pares se volverán a registrar limpiamente.
