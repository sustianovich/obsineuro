/*
 * Carga index.html y app.js en un DOM real y comprueba que la interfaz
 * arranca sin referencias nulas.
 *
 * Un `getElementById` que devuelve null no falla al cargar: falla más
 * tarde, cuando el usuario pulsa algo. Esta prueba lo detecta antes.
 *
 * Uso: node tests/test_ui_dom.mjs   (requiere: npm install jsdom)
 */

import {readFileSync} from "node:fs";
import {JSDOM} from "jsdom";

const html = readFileSync(
    new URL("../app/templates/index.html", import.meta.url),
    "utf-8"
);
const js = readFileSync(
    new URL("../app/static/app.js", import.meta.url),
    "utf-8"
);

let fallos = 0;

function check(nombre, condicion, detalle = "") {
    if (condicion) {
        console.log(`  OK   ${nombre}`);
    } else {
        fallos += 1;
        console.log(`  FALLO ${nombre} ${detalle}`);
    }
}

console.log("Interfaz");

const dom = new JSDOM(html, {
    runScripts: "outside-only",
    url: "http://127.0.0.1:8000/",
    pretendToBeVisual: true,
});
const {window} = dom;

// Sin red: se intercepta fetch para que la carga inicial no explote.
const llamadas = [];
window.fetch = (url) => {
    llamadas.push(String(url));
    return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
    });
};

const errores = [];
window.addEventListener("error", (event) => errores.push(event.message));

// 1. Todos los elementos que el script busca deben existir.
const pedidos = [
    ...js.matchAll(/getElementById\(\s*["']([^"']+)["']/g),
].map((match) => match[1]);
const ausentes = [
    ...new Set(pedidos.filter((id) => !window.document.getElementById(id))),
];
check(
    `${new Set(pedidos).size} elementos localizados`,
    ausentes.length === 0,
    ausentes.join(", ")
);

// 2. El script debe evaluarse sin lanzar.
let arranque = null;
try {
    window.eval(js);
} catch (error) {
    arranque = error;
}
check("el script se evalúa sin excepciones", arranque === null, arranque?.message ?? "");

// 3. Estructura resultante.
const doc = window.document;
check(
    "el panel de Conversaciones ya no está en la barra lateral",
    doc.querySelector(".sidebar .conversations-panel") === null
);
check(
    "el historial vive en el área de trabajo",
    doc.querySelector(".workspace .history-drawer") !== null
);
check(
    "el historial arranca plegado",
    doc.getElementById("history-drawer").open === false
);

// 4. El tema oscuro se puede activar desde la barra superior.
const themeToggle = doc.getElementById("theme-toggle");
check("el boton de fondo oscuro existe", themeToggle !== null);
themeToggle?.click();
check(
    "el boton activa el fondo oscuro",
    doc.documentElement.dataset.theme === "dark"
);
check(
    "el boton refleja el estado oscuro",
    themeToggle?.getAttribute("aria-pressed") === "true"
);

// 5. Todos los checkboxes en un único panel.
const casillas = [...doc.querySelectorAll('input[type="checkbox"]')];
const panelesConCasillas = new Set(
    casillas.map((casilla) => {
        const panel = casilla.closest("section");
        return panel ? panel.className : "(fuera de panel)";
    })
);
check(
    `los ${casillas.length} checkboxes están en un solo panel`,
    panelesConCasillas.size === 1,
    [...panelesConCasillas].join(" | ")
);
check(
    "ese panel es el de Opciones",
    [...panelesConCasillas][0] === "panel options-panel"
);
check(
    "el panel de Opciones es el último de la barra lateral",
    doc.querySelector(".sidebar section:last-of-type").className ===
        "panel options-panel"
);

// 5. Los cuatro ajustes esperados siguen presentes.
for (const id of [
    "expand-links",
    "project-verification-enabled",
    "project-memory-enabled",
    "use-memory",
]) {
    check(`ajuste #${id} disponible`, doc.getElementById(id) !== null);
}

// 6. Nada quedó huérfano: el historial conserva su lista y su vacío.
check(
    "la lista de conversaciones sigue existiendo",
    doc.querySelector(".history-drawer #conversation-list") !== null
);
check(
    "el mensaje de historial vacío sigue existiendo",
    doc.querySelector(".history-drawer #conversation-list-empty") !== null
);

// 7. Inspector de contexto: indicador accesible en la cabecera y sus
//    dos diálogos (conversación activa / turno histórico) presentes.
check(
    "el indicador de contexto existe en la cabecera",
    doc.getElementById("context-indicator") !== null
);
check(
    "el indicador anuncia que abre un diálogo",
    doc.getElementById("context-indicator")?.getAttribute("aria-haspopup")
        === "dialog"
);
check(
    "el diálogo del inspector de contexto existe",
    doc.getElementById("context-inspector-dialog") !== null
);
check(
    "el diálogo de detalle de turno existe",
    doc.getElementById("turn-inspector-dialog") !== null
);
check(
    "el panel del verificador y del redactor están en el inspector",
    doc.getElementById("ci-verifier") !== null
        && doc.getElementById("ci-writer") !== null
);

// 8. La interfaz nunca inserta contenido dinámico con innerHTML.
check(
    "app.js no usa innerHTML para insertar contenido",
    !js.includes("innerHTML")
);

check(
    "sin errores en tiempo de carga",
    errores.length === 0,
    errores.join(" | ")
);

console.log(
    fallos === 0 ? "\nInterfaz: correcta" : `\n${fallos} fallos`
);
process.exit(fallos === 0 ? 0 : 1);
