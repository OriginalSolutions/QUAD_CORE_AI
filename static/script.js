const chartDivPrice = document.getElementById('chart-price');
const chartDivPnL = document.getElementById('chart-pnl');
const overlay = document.getElementById('loading-overlay');
const statusSpan = document.getElementById('server-status');
let isChartInitialized = false;
let lastGraphTimestamp = 0; 
let globalPnlTimes = [];
let globalPnlValues = [];

function formatProb(probUp, w) {
    let cls = probUp > 50 ? 'up' : (probUp < 50 ? 'down' : 'neutral');
    let arrow = probUp > 50 ? '▲' : (probUp < 50 ? '▼' : '-');
    return `<span class="${cls}">${arrow} ${probUp.toFixed(1)}%</span> <span style="color:#555;font-size:9px;">(x${w.toFixed(2)})</span>`;
}

async function initChart() {
    try {
        const response = await fetch('/api/init');
        const data = await response.json();
        if (!data.history || data.history.length === 0) return;

        // --- PANEL AI ---
        if(data.models) {
            const m = data.models;
            const w = m.weights || {mc:1,rf:1,kan:1,net:1};
            
            if(m.config) {
                let t = m.config.temp !== undefined ? m.config.temp : 9.0;
                let s = m.config.sup !== undefined ? m.config.sup : 0.3;
                document.getElementById('mc-specs').innerText = `{Win:${m.config.win} Ahead:${m.config.ahead} Iter:${m.config.iter} T:${t} Sup:${s}}`;
            }
            if(m.rf_steps) {
                let diff = m.rf_steps.diff || 0;
                let trust = m.rf_steps.trust || 0;
                let op = (diff * trust) >= 0 ? "+" : "";
                document.getElementById('rf-math').innerText = `Raw:${m.rf_raw}% Acc:${m.rf_acc_view} ➜ 50% ${op} (${diff.toFixed(1)} × ${trust.toFixed(2)} [Trust])`;
            }
            document.getElementById('mc-val').innerHTML = formatProb(m.mc_prob, w.mc);
            document.getElementById('rf-val').innerHTML = formatProb(m.rf_prob, w.rf);
            document.getElementById('kan-val').innerHTML = formatProb(m.kan_prob, w.kan);
            document.getElementById('net-val').innerHTML = formatProb(m.neural_prob, w.net);
            
            // --- CONSENSUS ---
            let cVal = m.consensus_val;
            let stratMult = m.mult !== undefined ? m.mult : -1;
            let multText = stratMult > 0 ? "+1" : "-1";
            let cCls = cVal > 50.1 ? 'up' : (cVal < 49.9 ? 'down' : 'neutral');
            let cTxt = cVal > 50.1 ? 'BUY' : (cVal < 49.9 ? 'SELL' : 'NEUTRAL');
            
            document.getElementById('final-val').innerHTML = 
                `<span class="${cCls}">${cTxt} ${cVal.toFixed(1)}%</span> <span style="color:#666;font-size:16px;font-weight:bold;margin-left:8px;">(x${multText})</span>`;
            
            if(data.timestamps.length > 0) lastGraphTimestamp = data.timestamps[data.timestamps.length-1];
        }

        // --- WYKRES GÓRNY ---
        const traceReal = { x: data.dates, y: data.history, mode: 'lines', name: 'Price', line: { color: '#2979ff', width: 2 }, hovertemplate: '%{y:.0f}<extra>Price</extra>' };
        const traceStoch = { x: data.forecast_dates, y: data.stoch, mode: 'lines', name: 'Sim', line: { color: '#ff1744', width: 1, opacity: 0.7 }, hovertemplate: '%{y:.0f}<extra>Sim</extra>' };
        const traceTrend = { x: data.forecast_dates, y: data.trend, mode: 'lines', name: 'Trend', line: { color: '#ffea00', width: 2, dash: 'dash' }, hovertemplate: '%{y:.0f}<extra>Trend</extra>' };
        const traceRes = { x: data.forecast_dates, y: data.res, mode: 'lines', name: 'Res', line: { color: '#00e676', width: 1, dash: 'dot'}, hovertemplate: '%{y:.0f}<extra>Res</extra>' };
        const traceSup = { x: data.forecast_dates, y: data.sup, mode: 'lines', name: 'Sup', line: { color: '#d50000', width: 1, dash: 'dot'}, hovertemplate: '%{y:.0f}<extra>Sup</extra>' };
        
        Plotly.newPlot(chartDivPrice, [traceReal, traceStoch, traceTrend, traceRes, traceSup], {
            paper_bgcolor: '#0e0e0e', plot_bgcolor: '#0e0e0e', font: { color: '#9e9e9e' },
            margin: { t: 20, b: 30, l: 40, r: 60 }, hovermode: 'x unified',
            xaxis: { gridcolor: '#222', showgrid: true, range: [data.dates[0], data.forecast_dates[data.forecast_dates.length-1]] },
            yaxis: { side: 'right', gridcolor: '#222', showgrid: true, tickformat: '.0f' },
            legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0)' }
        }, {responsive: true});

        // --- WYKRES DOLNY (PnL) ---
        if(data.pnl && data.pnl.times.length > 0) {
            globalPnlTimes = data.pnl.times;
            globalPnlValues = data.pnl.balance;

            let globalMin = Math.min(...globalPnlValues);
            let globalMax = Math.max(...globalPnlValues);
            let sliderPad = (globalMax - globalMin) * 0.05; 
            if(sliderPad === 0) sliderPad = 50;

            // Trace 0: CZYSTA LINIA (dla suwaka)
            const traceForSlider = { 
                x: globalPnlTimes, y: globalPnlValues, mode: 'lines', 
                line: { color: '#00e676', width: 2 },
                showlegend: false, hoverinfo: 'skip'
            };

            // Trace 1: LINIA + ZNACZNIKI (dla głównego widoku)
            const traceMainPnL = { 
                x: globalPnlTimes, y: globalPnlValues, 
                mode: 'lines+markers', 
                name: 'Equity', 
                line: { color: '#00e676', width: 2 },
                marker: { size: 6, color: '#00e676' },
                hovertemplate: '%{y:.2f} USDT<extra></extra>',
                showlegend: false
            };

            const SHOW_LAST_N = 10;
            let xStart = globalPnlTimes.length > SHOW_LAST_N ? globalPnlTimes[globalPnlTimes.length - SHOW_LAST_N] : globalPnlTimes[0];
            let initSlice = globalPnlValues.slice(-SHOW_LAST_N);
            let yMin = Math.min(...initSlice), yMax = Math.max(...initSlice);
            let yPad = (yMax - yMin) * 0.1 || 50;

            const layoutPnL = {
                paper_bgcolor: '#0e0e0e', plot_bgcolor: '#0e0e0e', font: { color: '#9e9e9e' },
                margin: { t: 40, b: 30, l: 60, r: 80 }, hovermode: 'x unified',
                annotations: [{
                    text: "Virtual Balance USDT", xref: "paper", yref: "paper",
                    x: -0.06, y: 0.5, xanchor: 'center', yanchor: 'middle', textangle: -90, showarrow: false,
                    font: { size: 14, color: '#aaa' }
                }],
                xaxis: { 
                    gridcolor: '#222', range: [xStart, globalPnlTimes[globalPnlTimes.length-1]],
                    rangeslider: { 
                        visible: true, thickness: 0.15,
                        yaxis: { range: [globalMin - sliderPad, globalMax + sliderPad], rangemode: 'fixed' }
                    }
                }, 
                yaxis: { 
                    side: 'right', gridcolor: '#222', fixedrange: false, automargin: true, tickformat: '.0f',
                    range: [yMin - yPad, yMax + yPad]
                }
            };

            await Plotly.newPlot(chartDivPnL, [traceForSlider, traceMainPnL], layoutPnL, {responsive: true});

            chartDivPnL.on('plotly_relayout', function(ed){
                if(ed['xaxis.range'] || ed['xaxis.range[0]']) {
                    let xs = ed['xaxis.range'] ? ed['xaxis.range'][0] : ed['xaxis.range[0]'];
                    let xe = ed['xaxis.range'] ? ed['xaxis.range'][1] : ed['xaxis.range[1]'];
                    let ds = new Date(xs).getTime(), de = new Date(xe).getTime();
                    let minV = Infinity, maxV = -Infinity, f = false;
                    for(let i=0; i<globalPnlTimes.length; i++){
                        let t = new Date(globalPnlTimes[i]).getTime();
                        if(t >= ds && t <= de) {
                            let v = globalPnlValues[i];
                            if(v < minV) minV = v; if(v > maxV) maxV = v;
                            f = true;
                        }
                    }
                    if(f) {
                        let p = (maxV - minV) * 0.1 || 50;
                        Plotly.relayout(chartDivPnL, { 'yaxis.range': [minV - p, maxV + p] });
                    }
                }
            });
        }
        isChartInitialized = true;
    } catch (e) { console.error(e); }
}

async function updateLive() {
    try {
        const r = await fetch('/api/current_price');
        const d = await r.json();
        if(d.price) document.getElementById('last-price').innerText = d.price.toFixed(0);
        statusSpan.innerText = d.status || "Unknown";
        if(d.status === "TRAINING") overlay.style.visibility = "visible";
        else {
            if(overlay.style.visibility === "visible") initChart(); 
            overlay.style.visibility = "hidden";
        }
        if(d.closed_candle && isChartInitialized && d.closed_candle.ts > lastGraphTimestamp) {
            Plotly.extendTraces(chartDivPrice, { x: [[d.closed_candle.time]], y: [[d.closed_candle.price]] }, [0]);
            lastGraphTimestamp = d.closed_candle.ts;
        }
    } catch(e) {}
}
setInterval(updateLive, 1000);
initChart();