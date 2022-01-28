import React, { useState } from 'react';
import { useInterval } from 'usehooks-ts'
import TeX from '@matejmazur/react-katex'
import { calcOutflowSteps, d, FlowEdge, SvgDefs, Vertex } from "./DynFlowSvg";
import { Flow } from "./Flow";

import example2FlowData from "./example2FlowData.js"
import { Stepper } from 'spectacle';

const transitTime = [200, 200, 200]
const capacity = [10, 20, 20]

const flow = Flow.fromJson(example2FlowData)

const outflowSteps = flow.outflow.map(outflow => calcOutflowSteps(outflow, ['#a00', '#0a0']))

export const Example2Svg = () => {
    const sPos = [25, 150]
    const dxy = Math.sqrt(200**2 / 2) 
    const vPos = [25 + 100, 150 + Math.sqrt(200**2 - 100**2)]
    const tPos = [25 + 200, 150]
    return <Stepper values={[200]} alwaysVisible>
        {
            (value, step, isActive) => {
                const [t, setT] = useState(typeof value === 'number' ? value : 0)
                useInterval(() => {
                    setT(t => {
                        if (t < (value || 0)) return Math.min((value || 0), t + 0.5)
                        else if (t > (value || 0)) return Math.max((value || 0), t - 0.5)
                        return (value || 0)
                    })
                }, 1 / 30)
                const width = 450
                const height = vPos[1] + 25
                const svgIdPrefix = "example2-"
                return <div style={{ position: 'relative' }}>
                    <svg width={width} height={height}>
                        <SvgDefs svgIdPrefix={svgIdPrefix} />
                        <FlowEdge svgIdPrefix={svgIdPrefix} outflowSteps={outflowSteps[0]} from={sPos} to={tPos} capacity={capacity[0]} transitTime={transitTime[0]} queue={flow.queues[0]} t={t} />
                        <FlowEdge svgIdPrefix={svgIdPrefix}  outflowSteps={outflowSteps[1]} from={sPos} to={vPos} capacity={capacity[1]} transitTime={transitTime[1]} queue={flow.queues[1]} t={t} />
                        <FlowEdge svgIdPrefix={svgIdPrefix}  outflowSteps={outflowSteps[2]} from={vPos} to={tPos} capacity={capacity[2]} transitTime={transitTime[2]} queue={flow.queues[2]} t={t} />

                        <Vertex pos={sPos} label={<TeX>s</TeX>} />
                        <Vertex pos={vPos} label={<TeX>v</TeX>} />
                        <Vertex pos={tPos} label={<TeX>t</TeX>} />

                        <rect fill="white" x={vPos[0] - 25} y={0} width={50} height={50} rx={5} ry={5} stroke="lightgray" />
                        <PlayPause play={t !== value} x={vPos[0] - 10} y={5} size={20} />
                        <text x={vPos[0]} y={45} fontFamily='Open Sans, Sans Serif' fontSize={15} textAnchor='middle'>Time</text>
                    </svg>
                </div>
            }
        }
    </Stepper>
}


const PlayPause = ({ play, x, y, size }) => {
    const width = size / Math.sqrt(2)
    const pauseD = d.M(x + (size - width) / 2, y) + d.v(size) + d.l(width / 3, 0) + d.v(-size) + d.z
        + d.m(width * 2 / 3, 0) + d.v(size) + d.l(width / 3, 0) + d.v(-size) + d.z
    const playD = d.M(x + (size - width) / 2, y) + d.v(size) + d.l(width / 2, -size / 4) + d.v(-size / 2) + d.z
        + d.m(width / 2, size / 4) + d.v(size / 2) + d.l(width / 2, - size / 4) + d.v(0) + d.z
    return <path style={{ transition: "d 0.2s" }} d={play ? playD : pauseD} stroke="none" fill="gray" />
}