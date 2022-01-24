import React, { useState } from 'react';
import { useInterval } from 'usehooks-ts'
import TeX from '@matejmazur/react-katex'
import { BaseEdge, calcOutflowSteps, splitOutflowSteps, SvgDefs, Vertex } from "./DynFlowSvg";
import { Flow } from "./Flow";

import example1FlowData from "./example1FlowData.js"

const transitTime = [200, 200]
const capacity = [20, 10]

const flow = Flow.fromJson(example1FlowData)

const outflowSteps = flow.outflow.map(outflow => calcOutflowSteps(outflow, ['#a00', '#0a0']))

const FlowEdge = ({ from, to, outflowSteps, queue, t, capacity, transitTime }) => {
    const { inEdgeSteps, queueSteps } = splitOutflowSteps(outflowSteps, queue, transitTime, capacity, t)

    return <BaseEdge from={from} to={to} width={capacity} inEdgeSteps={inEdgeSteps} queueSteps={queueSteps} />
}

export const Example1Svg = () => {
    const sPos = [25, 200]
    const vPos = [225, 200]
    const tPos = [425, 200]
    const [t, setT] = useState(-100)
    useInterval(() => setT(t => t + 0.5), 1 / 30)
    return <svg width={500} height={500}>
        <SvgDefs />
        <FlowEdge outflowSteps={outflowSteps[0]} from={sPos} to={vPos} capacity={capacity[0]} transitTime={transitTime[0]} queue={flow.queues[0]} t={t} />
        <FlowEdge outflowSteps={outflowSteps[1]} from={vPos} to={tPos} capacity={capacity[1]} transitTime={transitTime[1]} queue={flow.queues[1]} t={t} />

        <Vertex pos={sPos} label={<TeX>s</TeX>} />
        <Vertex pos={tPos} label={<TeX>t</TeX>} />
        <Vertex pos={vPos} label={<TeX>v</TeX>} />
    </svg>
}
