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
                statusTxt.innerText = "Building 3D Model...";
                
                visualizeStructure(result.data);

            } catch (error) {
                console.error(error);
                statusTxt.innerHTML = `<span class="text-red-500">Connection Error. Check backend.</span>`;
            } finally {
                loader.classList.add('hidden');
            }
        }

        function visualizeStructure(layerData) {
            const traces = [];
            const blockHeight = 4; 
            const blockDepth = 4; 

            const textLabels = {
                x: [],
                y: [],
                z: [],
                text: []
            };

            const materialMap = new Map();

            layerData.forEach((layer) => {
                const zStart = layer.start_z;
                const zEnd = layer.start_z + layer.thickness;
                
                const color = layer.color || '#cccccc';
                const name = layer.name || 'Unknown';

                if (!materialMap.has(name)) {
                    materialMap.set(name, color);
                }

                // 1. Create Block Geometry
                const xMin = 0, xMax = blockDepth;
                const yMin = 0, yMax = blockHeight;
                
                const metaText = `
                    <b>${name}</b><br>
                    $\lambda$: ${layer.lamda}<br>
                    Thickness: ${layer.thickness} µm<br>
                    Ref Index: ${layer.ref_index}
                `;

                const meshTrace = {
                    type: "mesh3d",
                    x: [xMin, xMin, xMax, xMax, xMin, xMin, xMax, xMax],
                    y: [yMin, yMax, yMax, yMin, yMin, yMax, yMax, yMin],
                    z: [zStart, zStart, zStart, zStart, zEnd, zEnd, zEnd, zEnd],
                    i: [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j: [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k: [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    color: color,
                    opacity: 0.95,
                    flatshading: true,
                    name: name,
                    hoverinfo: 'text',
                    text: metaText
                };
                traces.push(meshTrace);

                const zMid = zStart + (layer.thickness / 2);
                textLabels.x.push(blockDepth / 2);
                textLabels.y.push(blockHeight / 2);
                textLabels.z.push(zMid);
                textLabels.text.push(`${layer.thickness} µm`);
            });

            const labelTrace = {
                type: 'scatter3d',
                mode: 'text',
                x: textLabels.x,
                y: textLabels.y,
                z: textLabels.z,
                text: textLabels.text,
                textfont: {
                    color: 'black',
                    size: 14,
                    family: 'Arial Black'
                },
                hoverinfo: 'none',
                showlegend: false
            };
            traces.push(labelTrace);
            
            const arrowTrace = {
                type: 'scatter3d',
                mode: 'lines+markers',
                x: [2, 2],
                y: [2, 2],
                z: [-1, 0],
                line: { color: 'red', width: 6 },
                marker: { symbol: 'arrow', size: 5, color: 'red'},
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
                margin: { t: 10, r: 10, l: 10, b: 10 },
                paper_bgcolor: 'white',
                scene: {
                    aspectmode: 'data', 
                    xaxis: { title: '', showgrid: false, zeroline: false, showticklabels: false, backgroundcolor: "white" },
                    yaxis: { title: '', showgrid: false, zeroline: false, showticklabels: false, backgroundcolor: "white" },
                    zaxis: { title: 'Sorted Layers (Z)', showgrid: true, gridcolor: '#eee' },
                    camera: {
                        eye: { x: 2.0, y: 1.5, z: 1.5 }, 
                        center: {x:0, y:0, z:0}
                    }
                },
                showlegend: false
            };

            Plotly.newPlot('outputPlot', traces, layout, {responsive: true});
            document.getElementById('statusText').innerText = "Structure Built (Sorted by λ).";
        }