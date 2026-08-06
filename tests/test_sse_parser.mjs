/*
 * Prueba del parser SSE de app.js sin navegador.
 *
 * Extrae readServerEvents del fichero real (no una copia) y lo somete a
 * los casos que de verdad ocurren en producción: eventos partidos entre
 * paquetes TCP, varios eventos en un mismo paquete y caracteres UTF-8
 * cortados por la mitad.
 *
 * Uso: node tests/test_sse_parser.mjs
 */

import {readFileSync} from "node:fs";

const source = readFileSync(
    new URL("../app/static/app.js", import.meta.url),
    "utf-8"
);

const start = source.indexOf("async function* readServerEvents");
const end = source.indexOf("function createStreamingTurn");
if (start === -1 || end === -1) {
    throw new Error("No se encontró readServerEvents en app.js");
}

const {readServerEvents} = await import(
    "data:text/javascript," +
    encodeURIComponent(
        source.slice(start, end) + "\nexport {readServerEvents};"
    )
);

function fakeResponse(packets) {
    const encoder = new TextEncoder();
    const chunks = packets.map((packet) =>
        typeof packet === "string" ? encoder.encode(packet) : packet
    );
    let index = 0;
    return {
        body: {
            getReader() {
                return {
                    read() {
                        if (index >= chunks.length) {
                            return Promise.resolve({
                                done: true,
                                value: undefined,
                            });
                        }
                        return Promise.resolve({
                            done: false,
                            value: chunks[index++],
                        });
                    },
                };
            },
        },
    };
}

async function collect(packets) {
    const events = [];
    for await (const event of readServerEvents(fakeResponse(packets))) {
        events.push(event);
    }
    return events;
}

let failures = 0;

function check(name, condition, detail = "") {
    if (condition) {
        console.log(`  OK   ${name}`);
    } else {
        failures += 1;
        console.log(`  FALLO ${name} ${detail}`);
    }
}

console.log("Parser SSE");

// 1. Evento simple.
{
    const events = await collect([
        'event: delta\ndata: {"text":"hola"}\n\n',
    ]);
    check(
        "evento único",
        events.length === 1 &&
            events[0].name === "delta" &&
            events[0].data.text === "hola"
    );
}

// 2. Varios eventos en un solo paquete.
{
    const events = await collect([
        'event: stage\ndata: {"stage":"writing"}\n\n' +
            'event: delta\ndata: {"text":"a"}\n\n' +
            'event: delta\ndata: {"text":"b"}\n\n',
    ]);
    check(
        "tres eventos en un paquete",
        events.length === 3 && events[2].data.text === "b"
    );
}

// 3. Evento partido entre paquetes.
{
    const events = await collect([
        "event: de",
        'lta\ndata: {"te',
        'xt":"partido"}',
        "\n\n",
    ]);
    check(
        "evento partido entre paquetes",
        events.length === 1 && events[0].data.text === "partido"
    );
}

// 4. Carácter UTF-8 cortado a mitad de byte.
{
    const encoder = new TextEncoder();
    const full = encoder.encode(
        'event: delta\ndata: {"text":"validación"}\n\n'
    );
    const cut = 30; // cae dentro de la ó multibyte
    const events = await collect([full.slice(0, cut), full.slice(cut)]);
    check(
        "UTF-8 partido entre paquetes",
        events.length === 1 && events[0].data.text === "validación",
        events.length ? JSON.stringify(events[0].data) : "sin eventos"
    );
}

// 5. JSON ilegible: se ignora sin romper el flujo.
{
    const originalError = console.error;
    console.error = () => {};
    const events = await collect([
        "event: delta\ndata: {roto\n\n" +
            'event: done\ndata: {"answer":"fin"}\n\n',
    ]);
    console.error = originalError;
    check(
        "evento corrupto no interrumpe el resto",
        events.length === 1 && events[0].name === "done"
    );
}

// 6. Flujo cortado sin terminador: no debe emitir un evento a medias.
{
    const events = await collect([
        'event: delta\ndata: {"text":"completo"}\n\n' +
            'event: delta\ndata: {"text":"incom',
    ]);
    check(
        "evento incompleto al cortarse la conexión",
        events.length === 1 && events[0].data.text === "completo"
    );
}

// 7. La telemetría enriquecida de contexto (modelo, ventanas, fuente)
//    llega intacta: el parser no debe recortar ni aplanar campos nuevos.
{
    const telemetry = {
        answer: "fin",
        agents: {
            model_context: {
                model: "qwen3.5:2b",
                profile_id: "balanced",
                model_context_window: 32768,
                source: "profile+ollama",
                verified: true,
                warning: null,
            },
            verifier: {
                status: "completed",
                model: "qwen3.5:2b",
                model_context_window: 32768,
                configured_context_window: 8192,
                effective_context_window: 8192,
                context_source: "profile+ollama",
                context_verified: true,
                prompt_tokens: 4200,
                completion_tokens: 520,
                total_tokens: 4720,
                remaining_tokens: 3472,
                usage_percent: 57.62,
                estimated: false,
            },
        },
    };
    const events = await collect([
        `event: done\ndata: ${JSON.stringify(telemetry)}\n\n`,
    ]);
    const verifier = events[0]?.data?.agents?.verifier;
    check(
        "la telemetría de contexto conserva todos sus campos",
        events.length === 1
            && verifier?.model === "qwen3.5:2b"
            && verifier?.model_context_window === 32768
            && verifier?.configured_context_window === 8192
            && verifier?.effective_context_window === 8192
            && verifier?.context_source === "profile+ollama"
            && verifier?.remaining_tokens === 3472
            && events[0].data.agents.model_context.source === "profile+ollama",
        JSON.stringify(events[0]?.data ?? {})
    );
}

console.log(failures === 0 ? "\nParser SSE: correcto" : `\n${failures} fallos`);
process.exit(failures === 0 ? 0 : 1);
