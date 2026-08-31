// ─── AnyLogic Web Toolkit ────────────────────────────────────────────────────

let anyLogicConnected = false;
let anyLogicListenersRegistered = false;

let selectorUbicacionActivo = false;
let selectorUbicacionSolicitudId = '';
let selectorUbicacionOcupado = false;

let gestionPedidosAnyLogicActiva = false;
let operacionPedidosWebEnCurso = false;
let operacionPedidosWebPendiente = null;

function webToolkitDisponible() {
    return (
        typeof window.AnyLogic !== 'undefined'
        && window.AnyLogic
        && typeof window.AnyLogic.call === 'function'
        && window.AnyLogic.events
        && typeof window.AnyLogic.events.on === 'function'
    );
}

function actualizarEstadoAnyLogic(texto, claseEstado) {
    const status = document.getElementById('anylogic-status');
    if (!status) return;

    status.textContent = texto;
    status.className = `connection-status ${claseEstado}`;
}

function registrarEventosAnyLogic() {
    if (!webToolkitDisponible() || anyLogicListenersRegistered) return;

    AnyLogic.events.on('pedemonteWebReady', payload => {
        anyLogicConnected = true;
        actualizarEstadoAnyLogic('AnyLogic conectado', 'status-connected');
        console.info('[PEDEMONTE WEB] Estado inicial recibido:', payload);
    });

    AnyLogic.events.on('pedemonteWebError', payload => {
        actualizarEstadoAnyLogic('Error de conexión', 'status-error');
        console.error('[PEDEMONTE WEB] Error informado por AnyLogic:', payload);
    });

    AnyLogic.events.on('pedemonteAbrirSelectorUbicacion', payload => {
        abrirSelectorUbicacion(payload);
    });

    AnyLogic.events.on('pedemonteUbicacionAceptada', payload => {
        if (!coincideSolicitudSelector(payload)) return;

        mostrarFeedbackSelector(
            'Ubicación aplicada correctamente en AnyLogic.',
            'selector-ok'
        );

        setTimeout(() => {
            cerrarSelectorUbicacionLocal();
        }, 120);
    });

    AnyLogic.events.on('pedemonteSelectorCancelado', payload => {
        if (!coincideSolicitudSelector(payload)) return;
        cerrarSelectorUbicacionLocal();
    });

    AnyLogic.events.on('pedemonteSelectorError', payload => {
        if (!coincideSolicitudSelector(payload)) return;

        selectorUbicacionOcupado = false;
        actualizarBotonesSelector();

        const mensaje = payload && payload.mensaje
            ? payload.mensaje
            : 'AnyLogic rechazó la ubicación seleccionada.';

        mostrarFeedbackSelector(mensaje, 'selector-error');
    });

    AnyLogic.events.on('pedemonteAbrirGestionPedidos', payload => {
        if (selectorUbicacionActivo) {
            cerrarSelectorUbicacionLocal();
        }

        gestionPedidosAnyLogicActiva = true;

        if (webToolkitDisponible() && AnyLogic.dialog) {
            AnyLogic.dialog.setTitle(
                'Gestión de pedidos — Pedemonte Digital Twin'
            );
        }

        mostrarFeedbackPedidos(
            payload && payload.mensaje
                ? payload.mensaje
                : 'Gestión conectada con AnyLogic.'
        );

        actualizarDisponibilidadGestionPedidos();
        actualizarBotonFinalizarGestion();
    });

    AnyLogic.events.on('pedemonteGestionPedidosFinalizada', payload => {
        cerrarGestionPedidosWebLocal(payload);
    });

    AnyLogic.events.on('pedemonteEstadoPedidosActualizado', payload => {
        aplicarEstadoPedidosAnyLogic(payload);
    });

    anyLogicListenersRegistered = true;
}

async function inicializarPuenteAnyLogic() {
    if (!webToolkitDisponible()) {
        anyLogicConnected = false;
        actualizarEstadoAnyLogic('Modo independiente', 'status-standalone');
        console.info('[PEDEMONTE WEB] AnyLogic Web Toolkit no está disponible.');
        return;
    }

    registrarEventosAnyLogic();
    actualizarEstadoAnyLogic('Conectando...', 'status-connecting');

    try {
        await AnyLogic.call('__ready__', {
            aplicacion: 'pedidos',
            contrato: 1,
            timestamp: new Date().toISOString()
        });

        anyLogicConnected = true;
        actualizarEstadoAnyLogic('AnyLogic conectado', 'status-connected');
        console.info('[PEDEMONTE WEB] Señal __ready__ confirmada.');
    } catch (error) {
        anyLogicConnected = false;
        actualizarEstadoAnyLogic('Error de conexión', 'status-error');
        console.error('[PEDEMONTE WEB] No se pudo completar __ready__:', error);
    }
}

function actualizarBotonFinalizarGestion() {
    const boton = document.getElementById('btn-finalizar-gestion');
    if (!boton) return;

    boton.classList.toggle(
        'is-hidden',
        !gestionPedidosAnyLogicActiva
    );

    boton.disabled = (
        !gestionPedidosAnyLogicActiva
        || operacionPedidosWebEnCurso
    );
}

function cerrarGestionPedidosWebLocal(payload) {
    gestionPedidosAnyLogicActiva = false;
    operacionPedidosWebEnCurso = false;
    operacionPedidosWebPendiente = null;

    const boton = document.getElementById('btn-finalizar-gestion');

    if (boton) {
        boton.disabled = false;
        boton.classList.add('is-hidden');
    }

    mostrarFeedbackPedidos(
        payload && payload.mensaje
            ? payload.mensaje
            : 'Carga web finalizada.',
        'feedback-ok'
    );

    actualizarDisponibilidadGestionPedidos();
}

async function finalizarGestionPedidosWeb() {
    if (!gestionPedidosAnyLogicActiva) {
        return;
    }

    if (operacionPedidosWebEnCurso) {
        mostrarFeedbackPedidos(
            'Espere a que termine la operación actual.',
            'feedback-working'
        );

        return;
    }

    if (!webToolkitDisponible()) {
        mostrarFeedbackPedidos(
            'La página no está conectada con AnyLogic.',
            'feedback-error'
        );

        return;
    }

    const requestId = crearRequestIdPedidos(
        'finalizar'
    );

    const boton = document.getElementById('btn-finalizar-gestion');

    if (boton) {
        boton.disabled = true;
    }

    mostrarFeedbackPedidos(
        'Finalizando la carga y volviendo a AnyLogic...',
        'feedback-working'
    );

    try {
        await AnyLogic.call(
            'finalizarGestionPedidosWeb',
            {
                requestId,
                timestamp: new Date().toISOString()
            }
        );
    } catch (error) {
        if (boton) {
            boton.disabled = false;
        }

        mostrarFeedbackPedidos(
            'No fue posible finalizar la gestión web.',
            'feedback-error'
        );

        console.error(
            '[PEDEMONTE WEB] Error al finalizar la gestión:',
            error
        );
    }
}

function crearRequestIdPedidos(prefijo) {
    return (
        prefijo
        + '-'
        + Date.now().toString(36)
        + '-'
        + Math.random().toString(36).slice(2, 9)
    );
}

function mostrarFeedbackPedidos(mensaje, clase = '') {
    const feedback = document.getElementById('pedido-action-feedback');
    if (!feedback) return;

    feedback.textContent = mensaje || '';
    feedback.className = (
        `pedido-action-feedback ${clase}`
    ).trim();
}

function normalizarPedidoAutoritativo(pedido) {
    const capacidad = Number(pedido && pedido.unidades_capacidad);
    const latitud = Number(pedido && pedido.latitud);
    const longitud = Number(pedido && pedido.longitud);
    const volcador = Number(pedido && pedido.requiere_volcador);

    return {
        pedido_id: String(pedido && pedido.pedido_id || '').trim(),
        turno: String(pedido && pedido.turno || 'MANANA').trim().toUpperCase(),
        latitud: Number.isFinite(latitud) ? latitud : 0,
        longitud: Number.isFinite(longitud) ? longitud : 0,
        unidades_capacidad: Number.isFinite(capacidad) ? capacidad : 1,
        requiere_volcador: volcador === 1 ? 1 : 0,
        hora_desde: String(pedido && pedido.hora_desde || '').trim(),
        hora_hasta: String(pedido && pedido.hora_hasta || '').trim(),
        cliente: String(pedido && pedido.cliente || ''),
        direccion: String(pedido && pedido.direccion || ''),
        barrio: String(pedido && pedido.barrio || ''),
        observaciones: String(pedido && pedido.observaciones || '')
    };
}

function reconstruirMarcadoresPedidos(ajustarVista = false) {
    if (!map) return;

    mapMarkers.forEach(item => {
        if (item && item.marker) {
            map.removeLayer(item.marker);
        }
    });

    mapMarkers = [];
    const bounds = [];

    pedidos.forEach(pedido => {
        if (
            !Number.isFinite(pedido.latitud)
            || !Number.isFinite(pedido.longitud)
        ) {
            return;
        }

        const marker = L.marker(
            [pedido.latitud, pedido.longitud],
            {
                icon: crearIcono(
                    pedido.pedido_id,
                    pedido.requiere_volcador
                )
            }
        ).addTo(map);

        marker.bindPopup(
            crearPopupHtml(pedido)
        );

        mapMarkers.push({
            id: pedido.pedido_id,
            marker
        });

        bounds.push([
            pedido.latitud,
            pedido.longitud
        ]);
    });

    if (ajustarVista && bounds.length > 0) {
        map.fitBounds(
            bounds,
            { padding: [40, 40] }
        );
    }
}

function restablecerFormularioWebDespuesDeAlta() {
    limpiarMarcadorTemporal();

    selectedLat = null;
    selectedLng = null;

    document.getElementById('coords-display').innerText = 'Lat: - | Lng: -';
    document.getElementById('selected-coords-status').innerText = 'Sin punto marcado';
    document.getElementById('selected-coords-status').style.color = '#f59e0b';

    document.getElementById('input-id').value = getNextDefaultId();
    document.getElementById('input-capacidad').value = '1';
    document.getElementById('input-volcador').checked = false;
    document.getElementById('time-desde').value = '';
    document.getElementById('time-hasta').value = '';
    document.getElementById('input-cliente').value = '';
    document.getElementById('input-direccion').value = '';
    document.getElementById('input-barrio').value = '';
    document.getElementById('input-observaciones').value = '';

    map.setView(INITIAL_CENTER, 14);
}

function actualizarDisponibilidadGestionPedidos() {
    const botonAgregar = document.getElementById('btn-agregar-pedido');
    const botonLimpiar = document.getElementById('btn-limpiar');
    const turno = document.getElementById('input-turno');
    const importar = document.querySelector(
        'button[onclick*="file-import"]'
    );

    if (botonAgregar) {
        botonAgregar.disabled = operacionPedidosWebEnCurso;
    }

    if (botonLimpiar) {
        botonLimpiar.disabled = (
            operacionPedidosWebEnCurso
            || pedidos.length === 0
        );
    }

    if (turno) {
        turno.disabled = (
            operacionPedidosWebEnCurso
            || (
                gestionPedidosAnyLogicActiva
                && pedidos.length > 0
            )
        );
    }

    if (importar) {
        importar.disabled = operacionPedidosWebEnCurso;
    }

    document.querySelectorAll('.pedido-item').forEach(item => {
        item.classList.toggle(
            'is-busy',
            operacionPedidosWebEnCurso
        );
    });

    /*
     * Los botones se renderizan mientras la operación todavía está
     * marcada como pendiente. Cuando llega el estado autoritativo de
     * AnyLogic, esta función debe volver a habilitarlos explícitamente.
     */
    document.querySelectorAll('.btn-delete').forEach(button => {
        button.disabled = operacionPedidosWebEnCurso;
    });

    actualizarBotonFinalizarGestion();
}

function aplicarEstadoPedidosAnyLogic(payload) {
    const lista = (
        payload
        && Array.isArray(payload.pedidos)
    )
        ? payload.pedidos
        : [];

    pedidos = lista
        .map(normalizarPedidoAutoritativo)
        .filter(pedido => pedido.pedido_id);

    const turnoAutoritativo = String(
        payload && payload.turno || ''
    ).trim().toUpperCase();

    if (
        turnoAutoritativo === 'MANANA'
        || turnoAutoritativo === 'TARDE'
    ) {
        document.getElementById('input-turno').value =
            turnoAutoritativo;
    }

    autoIdCounter = 1;
    document.getElementById('input-id').value =
        getNextDefaultId();

    reconstruirMarcadoresPedidos(
        Boolean(
            operacionPedidosWebPendiente
            && operacionPedidosWebPendiente.accion === 'IMPORTAR'
            && payload
            && payload.exito === true
        )
    );

    renderPedidosList();

    const requestId = String(
        payload && payload.requestId || ''
    );

    const coincidePendiente = (
        operacionPedidosWebPendiente
        && requestId
        && requestId === operacionPedidosWebPendiente.requestId
    );

    if (coincidePendiente) {
        const accion = operacionPedidosWebPendiente.accion;
        const exito = payload && payload.exito === true;
        const mensaje = String(
            payload && payload.mensaje
            || (
                exito
                    ? 'Operación completada.'
                    : 'AnyLogic rechazó la operación.'
            )
        );

        operacionPedidosWebEnCurso = false;
        operacionPedidosWebPendiente = null;

        mostrarFeedbackPedidos(
            mensaje,
            exito
                ? 'feedback-ok'
                : 'feedback-error'
        );

        if (exito && accion === 'AGREGAR') {
            restablecerFormularioWebDespuesDeAlta();
        }

        if (!exito) {
            alert(mensaje);
        }
    } else if (payload && payload.mensaje) {
        mostrarFeedbackPedidos(
            String(payload.mensaje),
            payload.exito === false
                ? 'feedback-error'
                : ''
        );
    }

    actualizarDisponibilidadGestionPedidos();
}

async function solicitarOperacionPedidosAnyLogic(
    tipoMensaje,
    accion,
    datos
) {
    if (!webToolkitDisponible() || !gestionPedidosAnyLogicActiva) {
        return false;
    }

    if (operacionPedidosWebEnCurso) {
        return false;
    }

    const requestId = crearRequestIdPedidos(
        accion.toLowerCase()
    );

    operacionPedidosWebEnCurso = true;
    operacionPedidosWebPendiente = {
        requestId,
        accion
    };

    mostrarFeedbackPedidos(
        'Validando la operación en AnyLogic...',
        'feedback-working'
    );

    actualizarDisponibilidadGestionPedidos();

    try {
        await AnyLogic.call(
            tipoMensaje,
            {
                requestId,
                ...datos,
                timestamp: new Date().toISOString()
            }
        );

        return true;
    } catch (error) {
        operacionPedidosWebEnCurso = false;
        operacionPedidosWebPendiente = null;

        actualizarDisponibilidadGestionPedidos();

        mostrarFeedbackPedidos(
            'No fue posible comunicarse con AnyLogic.',
            'feedback-error'
        );

        console.error(
            '[PEDEMONTE WEB] Falló la operación de pedidos:',
            error
        );

        return false;
    }
}


function normalizarNumero(valor) {
    const numero = Number(valor);
    return Number.isFinite(numero) ? numero : null;
}

function coincideSolicitudSelector(payload) {
    return (
        selectorUbicacionActivo
        && payload
        && String(payload.requestId || '') === selectorUbicacionSolicitudId
    );
}

function mostrarFeedbackSelector(mensaje, clase = '') {
    const feedback = document.getElementById('selector-feedback');
    if (!feedback) return;

    feedback.textContent = mensaje;
    feedback.className = `selector-feedback ${clase}`.trim();
}

function actualizarBotonesSelector() {
    const confirmar = document.getElementById('btn-confirmar-selector');
    const cancelar = document.getElementById('btn-cancelar-selector');

    if (confirmar) {
        confirmar.disabled = (
            selectorUbicacionOcupado
            || selectedLat === null
            || selectedLng === null
        );
    }

    if (cancelar) {
        cancelar.disabled = selectorUbicacionOcupado;
    }
}

function actualizarPanelSelector() {
    const coords = document.getElementById('selector-coords');

    if (coords) {
        coords.textContent = (
            selectedLat === null || selectedLng === null
                ? 'Lat: — | Lng: —'
                : `Lat: ${selectedLat} | Lng: ${selectedLng}`
        );
    }

    if (selectorUbicacionActivo && !selectorUbicacionOcupado) {
        mostrarFeedbackSelector(
            selectedLat === null || selectedLng === null
                ? 'Todavía no seleccionaste un punto.'
                : 'Punto listo para confirmar.'
        );
    }

    actualizarBotonesSelector();
}

function limpiarMarcadorTemporal() {
    if (tempMarker && map) {
        map.removeLayer(tempMarker);
    }

    tempMarker = null;
}

function establecerCoordenadasSeleccionadas(latitud, longitud, opciones = {}) {
    const lat = normalizarNumero(latitud);
    const lng = normalizarNumero(longitud);

    if (
        lat === null
        || lng === null
        || lat < -90
        || lat > 90
        || lng < -180
        || lng > 180
    ) {
        return false;
    }

    selectedLat = parseFloat(lat.toFixed(5));
    selectedLng = parseFloat(lng.toFixed(5));

    const coordsDisplay = document.getElementById('coords-display');
    const coordsStatus = document.getElementById('selected-coords-status');

    if (coordsDisplay) {
        coordsDisplay.innerText = `Lat: ${selectedLat} | Lng: ${selectedLng}`;
    }

    if (coordsStatus) {
        coordsStatus.innerText = 'Punto seleccionado';
        coordsStatus.style.color = '#10b981';
    }

    limpiarMarcadorTemporal();

    tempMarker = L.marker(
        [selectedLat, selectedLng],
        { opacity: 0.85 }
    ).addTo(map);

    tempMarker
        .bindPopup(
            `<b>Punto seleccionado</b><br>`
            + `Lat: ${selectedLat}<br>`
            + `Lng: ${selectedLng}`
        );

    if (opciones.abrirPopup !== false) {
        tempMarker.openPopup();
    }

    if (opciones.centrar === true) {
        map.setView(
            [selectedLat, selectedLng],
            opciones.zoom || 17
        );
    }

    actualizarPanelSelector();
    return true;
}

function abrirSelectorUbicacion(payload) {
    const requestId = String(payload && payload.requestId || '').trim();

    if (!requestId) {
        console.error('[PEDEMONTE WEB] Solicitud de ubicación sin requestId.');
        return;
    }

    selectorUbicacionActivo = true;
    selectorUbicacionSolicitudId = requestId;
    selectorUbicacionOcupado = false;

    document.body.classList.add('selector-mode');

    const panel = document.getElementById('location-selector-panel');
    if (panel) {
        panel.classList.remove('is-hidden');
        panel.setAttribute('aria-hidden', 'false');
    }

    if (webToolkitDisponible() && AnyLogic.dialog) {
        AnyLogic.dialog.setTitle('Seleccionar ubicación — Pedemonte Digital Twin');
    }

    const latitudInicial = normalizarNumero(payload && payload.latitudInicial);
    const longitudInicial = normalizarNumero(payload && payload.longitudInicial);
    const tieneInicial = Boolean(payload && payload.tieneUbicacionInicial);

    if (
        tieneInicial
        && latitudInicial !== null
        && longitudInicial !== null
    ) {
        establecerCoordenadasSeleccionadas(
            latitudInicial,
            longitudInicial,
            {
                centrar: true,
                zoom: Number(payload.zoom || 17),
                abrirPopup: false
            }
        );
    } else {
        selectedLat = null;
        selectedLng = null;
        limpiarMarcadorTemporal();

        map.setView(
            INITIAL_CENTER,
            Number(payload && payload.zoom || 14)
        );

        actualizarPanelSelector();
    }

    setTimeout(() => {
        if (map) map.invalidateSize();
    }, 100);
}

function cerrarSelectorUbicacionLocal() {
    selectorUbicacionActivo = false;
    selectorUbicacionSolicitudId = '';
    selectorUbicacionOcupado = false;

    document.body.classList.remove('selector-mode');

    const panel = document.getElementById('location-selector-panel');
    if (panel) {
        panel.classList.add('is-hidden');
        panel.setAttribute('aria-hidden', 'true');
    }

    selectedLat = null;
    selectedLng = null;
    limpiarMarcadorTemporal();
    actualizarPanelSelector();

    if (webToolkitDisponible() && AnyLogic.dialog) {
        AnyLogic.dialog.setTitle('Gestión de pedidos — Pedemonte Digital Twin');
    }
}

async function confirmarSelectorUbicacion() {
    if (
        !selectorUbicacionActivo
        || selectorUbicacionOcupado
        || selectedLat === null
        || selectedLng === null
    ) {
        return;
    }

    if (!webToolkitDisponible()) {
        mostrarFeedbackSelector(
            'La página no está conectada con AnyLogic.',
            'selector-error'
        );
        return;
    }

    selectorUbicacionOcupado = true;
    actualizarBotonesSelector();
    mostrarFeedbackSelector('Validando ubicación en AnyLogic...');

    try {
        await AnyLogic.call('ubicacionSeleccionada', {
            requestId: selectorUbicacionSolicitudId,
            latitud: selectedLat,
            longitud: selectedLng,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        selectorUbicacionOcupado = false;
        actualizarBotonesSelector();

        mostrarFeedbackSelector(
            'No fue posible enviar la ubicación a AnyLogic.',
            'selector-error'
        );

        console.error(
            '[PEDEMONTE WEB] Error al confirmar ubicación:',
            error
        );
    }
}

async function cancelarSelectorUbicacion() {
    if (!selectorUbicacionActivo || selectorUbicacionOcupado) return;

    if (!webToolkitDisponible()) {
        cerrarSelectorUbicacionLocal();
        return;
    }

    selectorUbicacionOcupado = true;
    actualizarBotonesSelector();
    mostrarFeedbackSelector('Cancelando selección...');

    try {
        await AnyLogic.call('selectorUbicacionCancelado', {
            requestId: selectorUbicacionSolicitudId,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        selectorUbicacionOcupado = false;
        actualizarBotonesSelector();

        mostrarFeedbackSelector(
            'No fue posible cancelar la selección.',
            'selector-error'
        );

        console.error(
            '[PEDEMONTE WEB] Error al cancelar selector:',
            error
        );
    }
}

const INITIAL_CENTER = [-32.8520, -60.7100];

let searchDebounce = null;

let map;
let tempMarker = null;
let selectedLat = null;
let selectedLng = null;
let pedidos = [];
let mapMarkers = [];
let autoIdCounter = 1;

// Genera el icono circular para un marcador de pedido.
function crearIcono(id, volcador) {
    const label = id.length > 4 ? id.substring(0, 4) : id;
    return L.divIcon({
        html: `<div class="custom-marker ${volcador === 1 ? 'volcador' : ''}">${label}</div>`,
        className: '',
        iconSize: [30, 30],
        iconAnchor: [15, 15]
    });
}

// Construye el HTML del popup de un pedido.
function crearPopupHtml(p) {
    return `
        <div style="font-family: sans-serif; padding: 4px; color: #1e293b;">
            <b style="color: #2563eb;">Pedido ${p.pedido_id}</b><br>
            <b>Turno:</b> ${p.turno}<br>
            <b>Capacidad:</b> ${p.unidades_capacidad}<br>
            <b>Volcador:</b> ${p.requiere_volcador === 1 ? 'Sí (1)' : 'No (0)'}<br>
            ${p.hora_desde ? `<b>Horario:</b> ${p.hora_desde} - ${p.hora_hasta}<br>` : ''}
            ${p.cliente ? `<b>Cliente:</b> ${p.cliente}<br>` : ''}
            ${p.direccion ? `<b>Dirección:</b> ${p.direccion}<br>` : ''}
            ${p.barrio ? `<b>Barrio:</b> ${p.barrio}<br>` : ''}
            ${p.observaciones ? `<b>Obs:</b> ${p.observaciones}<br>` : ''}
            <small style="color: #64748b;">Lat: ${p.latitud}, Lng: ${p.longitud}</small>
        </div>
    `;
}

function initMap() {
    map = L.map('map', {
        center: INITIAL_CENTER,
        zoom: 14,
        zoomControl: true
    });

    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }).addTo(map);

    map.on('click', function(e) {
        establecerCoordenadasSeleccionadas(
            e.latlng.lat,
            e.latlng.lng,
            {
                centrar: false,
                abrirPopup: true
            }
        );
    });
}

function getNextDefaultId() {
    let id = `P${String(autoIdCounter).padStart(3, '0')}`;
    while (pedidos.some(p => p.pedido_id === id)) {
        autoIdCounter++;
        id = `P${String(autoIdCounter).padStart(3, '0')}`;
    }
    return id;
}

async function agregarPedido() {
    if (selectedLat === null || selectedLng === null) {
        alert('Por favor, primero hacé clic en una ubicación del mapa.');
        return;
    }

    const idVal = document.getElementById('input-id').value.trim();
    const turnoVal = document.getElementById('input-turno').value;
    const capacidadVal = parseInt(
        document.getElementById('input-capacidad').value,
        10
    );
    const volcadorVal = (
        document.getElementById('input-volcador').checked
            ? 1
            : 0
    );
    const horaDesdeVal = document.getElementById('time-desde').value.trim();
    const horaHastaVal = document.getElementById('time-hasta').value.trim();
    const clienteVal = document.getElementById('input-cliente').value.trim();
    const direccionVal = document.getElementById('input-direccion').value.trim();
    const barrioVal = document.getElementById('input-barrio').value.trim();
    const obsVal = document.getElementById('input-observaciones').value.trim();

    if (!idVal) {
        alert('Ingresá un pedido_id válido.');
        return;
    }

    if (pedidos.some(p => p.pedido_id.toLowerCase() === idVal.toLowerCase())) {
        alert(`El pedido_id "${idVal}" ya existe. Debe ser único.`);
        return;
    }

    if (!Number.isInteger(capacidadVal) || capacidadVal <= 0) {
        alert('Las unidades de capacidad deben ser un entero mayor a 0.');
        return;
    }

    if (
        (horaDesdeVal && !horaHastaVal)
        || (!horaDesdeVal && horaHastaVal)
    ) {
        alert(
            'Las horas deben completarse ambas '
            + '(hora_desde y hora_hasta) o ninguna.'
        );
        return;
    }

    const pedido = {
        pedido_id: idVal,
        turno: turnoVal,
        latitud: selectedLat,
        longitud: selectedLng,
        unidades_capacidad: capacidadVal,
        requiere_volcador: volcadorVal,
        hora_desde: horaDesdeVal,
        hora_hasta: horaHastaVal,
        cliente: clienteVal,
        direccion: direccionVal,
        barrio: barrioVal,
        observaciones: obsVal
    };

    if (webToolkitDisponible() && gestionPedidosAnyLogicActiva) {
        await solicitarOperacionPedidosAnyLogic(
            'agregarPedidoWeb',
            'AGREGAR',
            { pedido }
        );

        return;
    }

    // Modo independiente: conserva el comportamiento original.
    pedidos.push(pedido);
    reconstruirMarcadoresPedidos(false);
    renderPedidosList();
    restablecerFormularioWebDespuesDeAlta();
}

function renderPedidosList() {
    const container = document.getElementById('pedidos-container');
    document.getElementById('count-pedidos').innerText = pedidos.length;
    document.getElementById('btn-exportar').disabled = pedidos.length === 0;

    if (pedidos.length === 0) {
        container.innerHTML = (
            '<div style="text-align: center; color: #64748b; '
            + 'padding: 20px; font-size: 0.85rem;">'
            + 'No hay puntos agregados todavía.</div>'
        );

        actualizarDisponibilidadGestionPedidos();
        return;
    }

    container.innerHTML = '';

    pedidos.forEach((p, index) => {
        const item = document.createElement('div');
        item.className = 'pedido-item';

        const badgeClass = (
            p.turno === 'MANANA'
                ? 'badge-manana'
                : 'badge-tarde'
        );

        item.innerHTML = `
            <div class="pedido-info">
                <div>
                    <span class="pedido-id">${p.pedido_id}</span>
                    <span class="badge-turno ${badgeClass}">${p.turno}</span>
                </div>
                <span class="pedido-details">
                    Cap: ${p.unidades_capacidad}
                    | Volcador: ${p.requiere_volcador}
                    ${p.hora_desde ? `| ${p.hora_desde}-${p.hora_hasta}` : ''}
                </span>
                <span
                    class="pedido-details"
                    style="font-family: monospace; font-size: 0.72rem;"
                >
                    (${p.latitud}, ${p.longitud})
                </span>
            </div>
            <button
                class="btn-delete"
                title="Eliminar pedido"
                onclick="eliminarPedido(${index})"
                ${operacionPedidosWebEnCurso ? 'disabled' : ''}
            >
                ✕
            </button>
        `;

        container.appendChild(item);
    });

    actualizarDisponibilidadGestionPedidos();
}

async function eliminarPedido(index) {
    const pedido = pedidos[index];
    if (!pedido || operacionPedidosWebEnCurso) return;

    if (
        !confirm(
            `¿Deseás eliminar el pedido ${pedido.pedido_id}?`
        )
    ) {
        return;
    }

    if (webToolkitDisponible() && gestionPedidosAnyLogicActiva) {
        await solicitarOperacionPedidosAnyLogic(
            'eliminarPedidoWeb',
            'ELIMINAR',
            {
                pedido_id: pedido.pedido_id
            }
        );

        return;
    }

    pedidos.splice(index, 1);
    reconstruirMarcadoresPedidos(false);
    renderPedidosList();
}

async function limpiarTodo() {
    if (pedidos.length === 0 || operacionPedidosWebEnCurso) return;

    if (!confirm('¿Deseás borrar todos los pedidos cargados?')) {
        return;
    }

    if (webToolkitDisponible() && gestionPedidosAnyLogicActiva) {
        await solicitarOperacionPedidosAnyLogic(
            'limpiarPedidosWeb',
            'LIMPIAR',
            {}
        );

        return;
    }

    pedidos = [];
    autoIdCounter = 1;
    reconstruirMarcadoresPedidos(false);
    document.getElementById('input-id').value = getNextDefaultId();
    renderPedidosList();
}

async function exportarExcel() {
    if (pedidos.length === 0) {
        alert('No hay pedidos para exportar.');
        return;
    }

    const botonExportar = document.getElementById('btn-exportar');

    if (botonExportar) {
        botonExportar.disabled = true;
    }

    mostrarFeedbackPedidos(
        'Preparando archivo Excel...',
        'feedback-working'
    );

    try {
        const data = pedidos.map(p => ({
            pedido_id:          p.pedido_id,
            turno:              p.turno,
            latitud:            p.latitud,
            longitud:           p.longitud,
            unidades_capacidad: p.unidades_capacidad,
            requiere_volcador:  p.requiere_volcador,
            hora_desde:         p.hora_desde || '',
            hora_hasta:         p.hora_hasta || '',
            cliente:            p.cliente || '',
            direccion:          p.direccion || '',
            barrio:             p.barrio || '',
            observaciones:      p.observaciones || ''
        }));

        const ws = XLSX.utils.json_to_sheet(data, {
            header: [
                'pedido_id',
                'turno',
                'latitud',
                'longitud',
                'unidades_capacidad',
                'requiere_volcador',
                'hora_desde',
                'hora_hasta',
                'cliente',
                'direccion',
                'barrio',
                'observaciones'
            ]
        });

        const wb = XLSX.utils.book_new();

        XLSX.utils.book_append_sheet(
            wb,
            ws,
            'Pedidos'
        );

        /*
         * Dentro de WebToolkit no usamos XLSX.writeFile().
         * Esa función intenta iniciar una descarga normal del navegador,
         * pero la ventana JCEF no registra un DownloadHandler.
         *
         * En su lugar, generamos el XLSX en Base64 y utilizamos la API
         * nativa de archivos que WebToolkit expone mediante AnyLogic.files.
         */
        if (
            webToolkitDisponible()
            &&
            AnyLogic.files
            &&
            typeof AnyLogic.files.saveDialog === 'function'
            &&
            typeof AnyLogic.files.write === 'function'
        ) {
            let ruta = await AnyLogic.files.saveDialog(
                'Exportar pedidos',
                'pedidos_hoy.xlsx'
            );

            if (!ruta) {
                mostrarFeedbackPedidos(
                    'Exportación cancelada.'
                );

                return;
            }

            ruta = String(ruta);

            if (!ruta.toLowerCase().endsWith('.xlsx')) {
                ruta += '.xlsx';
            }

            const contenidoBase64 = XLSX.write(
                wb,
                {
                    bookType: 'xlsx',
                    type: 'base64'
                }
            );

            await AnyLogic.files.write(
                ruta,
                contenidoBase64,
                true
            );

            mostrarFeedbackPedidos(
                'Archivo Excel exportado correctamente.',
                'feedback-ok'
            );

            return;
        }

        /*
         * Fuera de AnyLogic conservamos la descarga tradicional
         * para que la página siga funcionando en un navegador común.
         */
        XLSX.writeFile(
            wb,
            'pedidos_hoy.xlsx'
        );

        mostrarFeedbackPedidos(
            'Archivo Excel exportado correctamente.',
            'feedback-ok'
        );

    } catch (error) {
        console.error(
            '[PEDEMONTE WEB] Error al exportar Excel:',
            error
        );

        mostrarFeedbackPedidos(
            'No fue posible exportar el archivo Excel.',
            'feedback-error'
        );

        alert(
            'No fue posible exportar el archivo Excel.'
        );

    } finally {
        if (botonExportar) {
            botonExportar.disabled =
                pedidos.length === 0;
        }
    }
}

function importarExcel(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = async function(e) {
        try {
            const data = new Uint8Array(e.target.result);
            const wb = XLSX.read(data, { type: 'array' });

            const sheetName = wb.SheetNames.includes('Pedidos')
                ? 'Pedidos'
                : wb.SheetNames[0];

            const rows = XLSX.utils.sheet_to_json(
                wb.Sheets[sheetName],
                { defval: '' }
            );

            if (rows.length === 0) {
                alert('El archivo no contiene filas de datos.');
                return;
            }

            const getVal = (row, ...cols) => {
                for (const col of cols) {
                    const key = Object.keys(row).find(
                        k => k.trim().toLowerCase() === col.toLowerCase()
                    );

                    if (key !== undefined) {
                        return String(row[key]).trim();
                    }
                }

                return '';
            };

            const pedidosImportados = [];

            rows.forEach((row, idx) => {
                const pLat = parseFloat(
                    getVal(row, 'latitud', 'lat')
                );

                const pLng = parseFloat(
                    getVal(row, 'longitud', 'lng', 'lon')
                );

                if (
                    !Number.isFinite(pLat)
                    || !Number.isFinite(pLng)
                    || pLat < -90
                    || pLat > 90
                    || pLng < -180
                    || pLng > 180
                ) {
                    throw new Error(
                        `Fila ${idx + 2}: coordenadas inválidas.`
                    );
                }

                const pId = (
                    getVal(row, 'pedido_id', 'id')
                    || `P_IMP_${idx + 1}`
                );

                const turnoTexto = getVal(row, 'turno').toUpperCase();
                const pTurno = turnoTexto === 'TARDE'
                    ? 'TARDE'
                    : 'MANANA';

                const pCap = parseInt(
                    getVal(
                        row,
                        'unidades_capacidad',
                        'capacidad',
                        'tamanio'
                    ) || '1',
                    10
                );

                if (!Number.isInteger(pCap) || pCap <= 0) {
                    throw new Error(
                        `Fila ${idx + 2}: capacidad inválida.`
                    );
                }

                const pVolc = ['1', 'true', 'si', 'sí'].includes(
                    getVal(
                        row,
                        'requiere_volcador',
                        'volcador'
                    ).toLowerCase()
                )
                    ? 1
                    : 0;

                pedidosImportados.push({
                    pedido_id: pId,
                    turno: pTurno,
                    latitud: pLat,
                    longitud: pLng,
                    unidades_capacidad: pCap,
                    requiere_volcador: pVolc,
                    hora_desde: getVal(
                        row,
                        'hora_desde',
                        'inicio_min'
                    ),
                    hora_hasta: getVal(
                        row,
                        'hora_hasta',
                        'fin_min'
                    ),
                    cliente: getVal(row, 'cliente'),
                    direccion: getVal(row, 'direccion'),
                    barrio: getVal(row, 'barrio'),
                    observaciones: getVal(row, 'observaciones')
                });
            });

            if (webToolkitDisponible() && gestionPedidosAnyLogicActiva) {
                await solicitarOperacionPedidosAnyLogic(
                    'importarPedidosWeb',
                    'IMPORTAR',
                    {
                        pedidos: pedidosImportados
                    }
                );

                return;
            }

            pedidos = pedidosImportados;
            reconstruirMarcadoresPedidos(true);
            renderPedidosList();

            alert(
                `Se importaron ${pedidos.length} pedidos correctamente.`
            );
        } catch (error) {
            console.error(error);
            alert(
                error && error.message
                    ? error.message
                    : 'Ocurrió un error al leer el archivo Excel.'
            );
        } finally {
            event.target.value = '';
        }
    };

    reader.readAsArrayBuffer(file);
}

// Busqueda de direcciones via Nominatim (OSM), restringida al viewport actual del mapa.
async function buscarDireccion() {
    const query = document.getElementById('search-input').value.trim();
    const resultsEl = document.getElementById('search-results');

    if (!query) {
        resultsEl.innerHTML = '';
        return;
    }

    resultsEl.innerHTML = '<div class="search-status">Buscando...</div>';

    try {
        const b = map.getBounds();
        const params = new URLSearchParams({
            q: query,
            format: 'json',
            limit: 5,
            viewbox: `${b.getWest()},${b.getNorth()},${b.getEast()},${b.getSouth()}`,
            bounded: 1,
            'accept-language': 'es'
        });

        const res = await fetch(`https://nominatim.openstreetmap.org/search?${params}`);
        const results = await res.json();

        if (results.length === 0) {
            resultsEl.innerHTML = '<div class="search-status">Sin resultados en el área visible.</div>';
            return;
        }

        resultsEl.innerHTML = '';
        results.forEach(r => {
            const item = document.createElement('div');
            item.className = 'search-result-item';
            item.textContent = r.display_name;
            item.onclick = () => seleccionarResultado(parseFloat(r.lat), parseFloat(r.lon));
            resultsEl.appendChild(item);
        });

    } catch (err) {
        console.error(err);
        resultsEl.innerHTML = '<div class="search-status">Error al conectar con el servicio de búsqueda.</div>';
    }
}

function seleccionarResultado(lat, lng) {
    document.getElementById('search-results').innerHTML = '';
    document.getElementById('search-input').value = '';

    establecerCoordenadasSeleccionadas(
        lat,
        lng,
        {
            centrar: true,
            zoom: 17,
            abrirPopup: true
        }
    );
}

window.onload = function() {
    initMap();
    document.getElementById('input-id').value = getNextDefaultId();

    document.getElementById('search-input').addEventListener('keydown', e => {
        if (e.key === 'Enter') buscarDireccion();
    });

    renderPedidosList();
    actualizarDisponibilidadGestionPedidos();
    inicializarPuenteAnyLogic();
};
