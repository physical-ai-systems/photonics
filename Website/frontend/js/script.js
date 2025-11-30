  document.addEventListener('DOMContentLoaded', () => {
            updateInputPlot();
            const table = document.getElementById('inputTable');
            if (table) {
                const handler = (e) => {
                    if (e.target && (e.target.matches('.input-r') || e.target.matches('.input-lambda'))) {
                        updateInputPlot();
                    }
                };
                table.addEventListener('input', handler, true);
                table.addEventListener('change', handler, true);
            }
        });

        function addColumn() {
            const headerRow = document.querySelector('#inputTable thead tr');
            const bodyRows = document.querySelectorAll('#inputTable tbody tr');
            const count = headerRow.children.length; 
            
            const th = document.createElement('th');
            th.className = "py-2 text-center";
            th.innerText = `${count}`;
            headerRow.appendChild(th);

            const tdLambda = document.createElement('td');
            tdLambda.className = "p-1";
            tdLambda.innerHTML = '<input type="number" value="2.0" step="0.1" class="w-full bg-slate-50 border rounded px-2 py-1 text-center focus:ring-2 focus:ring-blue-400 outline-none input-lambda">';
            bodyRows[0].appendChild(tdLambda);

            const tdR = document.createElement('td');
            tdR.className = "p-1";
            tdR.innerHTML = '<input type="number" value="0.5" min="0" max="1" step="0.1" class="w-full bg-slate-50 border rounded px-2 py-1 text-center focus:ring-2 focus:ring-purple-400 outline-none input-r">';
            bodyRows[1].appendChild(tdR);
            updateInputPlot();
        }

        function removeColumn() {
            const headerRow = document.querySelector('#inputTable thead tr');
            const bodyRows = document.querySelectorAll('#inputTable tbody tr');
            if (headerRow.children.length > 2) {
                headerRow.removeChild(headerRow.lastElementChild);
                bodyRows[0].removeChild(bodyRows[0].lastElementChild);
                bodyRows[1].removeChild(bodyRows[1].lastElementChild);
            }
            updateInputPlot();
        }

        function getData() {
            const lambdas = Array.from(document.querySelectorAll('.input-lambda')).map(i => parseFloat(i.value));
            const rs = Array.from(document.querySelectorAll('.input-r')).map(i => parseFloat(i.value));
            
            const errorMsg = document.getElementById('error-msg');
            const hasInvalidR = rs.some(r => r < 0 || r > 1 || isNaN(r));
            
            if (hasInvalidR) {
                errorMsg.classList.remove('hidden');
                return null;
            }
            errorMsg.classList.add('hidden');
            
            return { lamda: lambdas, r: rs };
        }

        function updateInputPlot() {
            const data = getData();
            if (!data) return;

            // Sort by lambda values
            const sortedIndices = data.lamda
                .map((val, idx) => ({val, idx}))
                .sort((a, b) => a.val - b.val)
                .map(item => item.idx);
            
            const sortedLamda = sortedIndices.map(i => data.lamda[i]);
            const sortedR = sortedIndices.map(i => data.r[i]);

            const trace = {
                x: sortedLamda,
                y: sortedR,
                mode: 'lines+markers',
                type: 'scatter',
                line: { shape: 'spline', color: '#4f46e5', width: 2 },
                marker: { size: 6, color: '#ec4899' },
                name: 'R vs λ'
            };

            const layout = {
                margin: { t: 20, r: 20, l: 40, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: 'λ (nm)', showgrid: true, gridcolor: '#e2e8f0' },
                yaxis: { title: 'R (A.U.)', range: [0, 1.1], showgrid: true, gridcolor: '#e2e8f0' }
            };

            Plotly.newPlot('inputPlot', [trace], layout, {displayModeBar: false});
            
            const producedPlot = document.getElementById('producedPlot');
            if (producedPlot) {
                if (producedPlot._fullLayout) {
                    Plotly.purge('producedPlot');
                }
                producedPlot.innerHTML = '<span class="text-sm">No data yet</span>';
            }
        }

        async function sendData() {
            const data = getData();
            if (!data) return;

            const statusTxt = document.getElementById('statusText');
            const loader = document.getElementById('loading');
            
            statusTxt.innerText = "Computing layers...";
            loader.classList.remove('hidden');

            try {
                const response = await fetch('http://127.0.0.1:8000/calculate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                if (!response.ok) throw new Error("Server error");

                const result = await response.json();
                statusTxt.innerText = "Building Structure...";
                
                let layers = result.data;
                let producedSpectrum = null;
                
                if (result.data.layers) {
                    layers = result.data.layers;
                    producedSpectrum = result.data.produced_spectrum;
                }
                
                visualizeStructure(layers);
                
                // Reset plot to remove old produced spectrum
                updateInputPlot();

                if (producedSpectrum) {
                    plotProducedSpectrum(producedSpectrum);
                }

            } catch (error) {
                console.error(error);
                statusTxt.innerHTML = `<span class="text-red-500">Connection Error. Check backend.</span>`;
            } finally {
                loader.classList.add('hidden');
            }
        }

        function plotProducedSpectrum(spectrumData) {
            const trace = {
                x: spectrumData.lamda,
                y: spectrumData.r,
                mode: 'lines',
                type: 'scatter',
                line: { shape: 'spline', color: 'red', width: 2 },
                name: 'Produced Spectrum'
            };
            
            Plotly.addTraces('inputPlot', trace);

            const layout = {
                margin: { t: 20, r: 20, l: 40, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: 'λ (nm)', showgrid: true, gridcolor: '#e2e8f0' },
                yaxis: { title: 'R (A.U.)', range: [0, 1.1], showgrid: true, gridcolor: '#e2e8f0' }
            };
            
            const producedPlot = document.getElementById('producedPlot');
            if (producedPlot) {
                Plotly.newPlot('producedPlot', [trace], layout, {displayModeBar: false});
            }
        }

        function visualizeStructure(layerData) {
            const traces = [];
            const blockWidth = 1; 

            const textLabels = {
                x: [],
                y: [],
                text: []
            };
            
            const hoverPoints = {
                x: [],
                y: [],
                text: [],
                color: []
            };

            const shapes = [];
            const materialMap = new Map();

            layerData.forEach((layer) => {
                const zStart = layer.start_z;
                const zEnd = layer.start_z + layer.thickness;
                const zMid = zStart + (layer.thickness / 2);
                
                const color = layer.color || '#cccccc';
                const name = layer.name || 'Unknown';

                if (!materialMap.has(name)) {
                    materialMap.set(name, color);
                }

                // Create 2D Rectangle Shape
                shapes.push({
                    type: 'rect',
                    x0: 0,
                    y0: zStart,
                    x1: blockWidth,
                    y1: zEnd,
                    fillcolor: color,
                    line: {
                        color: 'rgba(0,0,0,0.1)',
                        width: 1
                    }
                });

                const metaText = `
                    <b>${name}</b><br>
                    λ: ${layer.lamda}<br>
                    Thickness: ${layer.thickness} µm<br>
                    Ref Index: ${layer.ref_index}
                `;
                
                // Collect hover data
                hoverPoints.x.push(blockWidth / 2);
                hoverPoints.y.push(zMid);
                hoverPoints.text.push(metaText);
                hoverPoints.color.push(color);

                // Collect label data
                textLabels.x.push(blockWidth / 2);
                textLabels.y.push(zMid);
                textLabels.text.push(`${layer.thickness} µm`);
            });

            // Trace for Hover Info (invisible points)
            const hoverTrace = {
                x: hoverPoints.x,
                y: hoverPoints.y,
                mode: 'markers',
                marker: { opacity: 0 }, // Invisible markers
                text: hoverPoints.text,
                hoverinfo: 'text',
                showlegend: false
            };
            traces.push(hoverTrace);

            // Trace for Text Labels
            const labelTrace = {
                mode: 'text',
                x: textLabels.x,
                y: textLabels.y,
                text: textLabels.text,
                textfont: {
                    color: 'black',
                    size: 12,
                    family: 'Arial'
                },
                hoverinfo: 'none',
                showlegend: false
            };
            traces.push(labelTrace);
            
            // Incident Light Arrow (2D)
            const arrowTrace = {
                x: [blockWidth / 2, blockWidth / 2],
                y: [-1, 0], // Assuming structure starts at 0
                mode: 'lines+markers',
                line: { color: 'red', width: 4 },
                marker: { symbol: 'arrow-up', size: 10, color: 'red'},
                name: 'Incident Light',
                hoverinfo: 'name'
            };
            traces.push(arrowTrace);

            const legendContainer = document.getElementById('dynamicLegend');
            legendContainer.innerHTML = ''; 
            
            materialMap.forEach((color, name) => {
                const item = document.createElement('div');
                item.className = "flex items-center gap-2 px-3 py-1 rounded border bg-white shadow-sm";
                item.innerHTML = `
                    <div class="w-4 h-4 rounded-sm" style="background-color: ${color}; border: 1px solid rgba(0,0,0,0.1);"></div>
                    <span class="text-sm font-medium text-slate-700">${name}</span>
                `;
                legendContainer.appendChild(item);
            });

            const layout = {
                title: '',
                margin: { t: 20, r: 20, l: 50, b: 40 },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                xaxis: { 
                    title: '', 
                    showgrid: false, 
                    zeroline: false, 
                    showticklabels: false, 
                    range: [-0.5, 1.5] // Center the block
                },
                yaxis: { 
                    title: 'Position (µm)', 
                    showgrid: true, 
                    gridcolor: '#eee',
                    zeroline: true
                },
                shapes: shapes,
                showlegend: false,
                hovermode: 'closest'
            };

            Plotly.newPlot('outputPlot', traces, layout, {responsive: true});
            document.getElementById('statusText').innerText = "Structure Built (Sorted by λ).";
        }