import React, { useState } from 'react';
import { useInterval } from 'usehooks-ts'
import TeX from '@matejmazur/react-katex'
import { BaseEdge, calcOutflowSteps, d, splitOutflowSteps, SvgDefs, Vertex } from "./DynFlowSvg";
import { Flow } from "./Flow";

import example1FlowData from "./example1FlowData.js"
import { Stepper } from 'spectacle';

const transitTime = [200, 200, 200]
const capacity = [20, 10, 10]

const flow = Flow.fromJson(example1FlowData)

const outflowSteps = flow.outflow.map(outflow => calcOutflowSteps(outflow, ['#a00', '#0a0']))

const FlowEdge = ({ from, to, outflowSteps, queue, t, capacity, transitTime, visible = true }) => {
    const { inEdgeSteps, queueSteps } = splitOutflowSteps(outflowSteps, queue, transitTime, capacity, t)

    return <BaseEdge visible={visible} from={from} to={to} width={capacity} inEdgeSteps={inEdgeSteps} queueSteps={queueSteps} />
}

export const Example1Svg = () => {
    const sPos = [25, 150]
    const vPos = [225, 150]
    const tPos = [425, 150]
    const s2Pos = [225 - 200 / Math.sqrt(2), 150 + 200 / Math.sqrt(2)]
    return <Stepper values={[0, 400, 800, 1200, 1200.00001, 1450, 1900]} alwaysVisible>
        {
            (value, step, isActive) => {
                const [t, setT] = useState(typeof value === 'number' ? value : 0)
                useInterval(() => {
                    setT(t => {
                        if (t < value) return Math.min(value, t + 0.5)
                        else if (t > value) return Math.max(value, t - 0.5)
                        return value
                    })
                }, 1 / 30)
                const width = 450
                const height = s2Pos[1] + 25
                return <div style={{ position: 'relative' }}>
                    <svg width={width} height={height}>
                        <SvgDefs />
                        <FlowEdge outflowSteps={outflowSteps[0]} from={sPos} to={vPos} capacity={capacity[0]} transitTime={transitTime[0]} queue={flow.queues[0]} t={t} />
                        <FlowEdge outflowSteps={outflowSteps[1]} from={vPos} to={tPos} capacity={capacity[1]} transitTime={transitTime[1]} queue={flow.queues[1]} t={t} />
                        <FlowEdge visible={t > 1200} outflowSteps={outflowSteps[2]} from={s2Pos} to={vPos} capacity={capacity[2]} transitTime={transitTime[2]} queue={flow.queues[2]} t={t} />

                        <Vertex pos={sPos} label={<TeX>s_1</TeX>} />
                        <Vertex pos={tPos} label={<TeX>t_1</TeX>} />
                        <Vertex pos={vPos} label={<TeX>v</TeX>} />
                        <Vertex pos={s2Pos} label={<TeX>s_2</TeX>} visible={t > 1200} />

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