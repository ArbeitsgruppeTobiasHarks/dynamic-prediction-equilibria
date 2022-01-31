import React from 'react'
import { d, ForeignObjectLabel } from "./DynFlowSvg"
import { Tex } from './tex'

const height = 120
const width = 220
const padding = 20
const origin = [padding, height - padding]
const bartheta = origin[0] + 50 + 10 + 5
const qBartheta = origin[1] + -50 + 0 + 2.5

const H = 70
const delta = 25
const qBarthetaMinusDelta = origin[1] + - (bartheta - delta - origin[0])

export const ConstantPredictorSvg = ({ minimize }) => {
    const predictedPath = <path d={d.M(bartheta, qBartheta) + d.H(width - padding)}
        stroke='red' strokeDasharray={5} />
    return <Diagram minimize={minimize} predictedPath={predictedPath} />
}

export const ZeroPredictorSvg = ({ minimize }) => {
    const predictedPath = <path d={d.M(bartheta, origin[1]) + d.H(width - padding)}
        stroke='red' strokeDasharray={5} />
    return <Diagram predictedPath={predictedPath} minimize={minimize} />
}

const BarthetaPlusH = () => <>
    <line stroke="lightgray" x1={bartheta + H} y1={padding} x2={bartheta + H} y2={height - padding} />
    <ForeignObjectLabel cx={bartheta + H} cy={10} width={60}>
        <span style={{ fontSize: '0.7em' }}>{Tex`\hspace{0.35em}\bar\theta {+} H`}</span>
    </ForeignObjectLabel>
</>

const BarthetaMinusDelta = () => <>
    <line stroke="lightgray" x1={bartheta - delta} y1={padding} x2={bartheta - delta} y2={height - padding} />
    <ForeignObjectLabel cx={bartheta - delta} cy={10} width={60}>
        <span style={{ fontSize: '0.7em' }}>{Tex`\bar\theta {-} \delta`}</span>
    </ForeignObjectLabel>
</>

export const LinearPredictorSvg = ({ minimize }) => {
    const predictedPath = <>
        <BarthetaPlusH />
        <path d={d.M(bartheta, qBartheta) + d.l(H, H / 2) + d.H(width - padding)} stroke='red' strokeDasharray={5} />
    </>
    return <Diagram minimize={minimize} predictedPath={predictedPath} />
}

export const RegularizedLinearPredictorSvg = ({ minimize }) => {
    const m = (qBartheta - qBarthetaMinusDelta) / delta
    const predictedPath = <>
        <BarthetaPlusH />
        <BarthetaMinusDelta />
        <path d={d.M(bartheta - delta, qBarthetaMinusDelta) + d.L(bartheta, qBartheta)} stroke='gray' strokeDasharray={5} />
        <path d={d.M(bartheta, qBartheta) + d.l(H, m * H) + d.H(width - padding)} stroke='red' strokeDasharray={5} />    </>
    return <Diagram minimize={minimize} predictedPath={predictedPath} />
}

export const RegressionPredictorSvg = ({ minimize }) => {
    const delta2 = H / 4
    const predictedPath = <>
        <line stroke="lightgray" x1={bartheta - 3 * delta2} y1={padding} x2={bartheta - 3 * delta2} y2={height - padding} />
        <line stroke="lightgray" x1={bartheta - 2 * delta2} y1={padding} x2={bartheta - 2 * delta2} y2={height - padding} />
        <line stroke="lightgray" x1={bartheta - delta2} y1={padding} x2={bartheta - delta2} y2={height - padding} />
        <line stroke="lightgray" x1={bartheta + delta2} y1={padding} x2={bartheta + delta2} y2={height - padding} />
        <line stroke="lightgray" x1={bartheta + 2 * delta2} y1={padding} x2={bartheta + 2 * delta2} y2={height - padding} />
        <line stroke="lightgray" x1={bartheta + 3 * delta2} y1={padding} x2={bartheta + 3 * delta2} y2={height - padding} />
        <line stroke="lightgray" x1={bartheta + 4 * delta2} y1={padding} x2={bartheta + 4 * delta2} y2={height - padding} />
        <path d={d.M(bartheta, qBartheta) + d.l(delta2, 5) + d.l(delta2, 15) + d.l(delta2, -10) + d.l(delta2, -20) + d.H(width - padding)} stroke='red' strokeDasharray={5} />
        <ForeignObjectLabel cx={bartheta - 1.5 * delta2} cy={15} width={300} height={300}>
            <div style={{ fontSize: "0.7em" }}>{Tex`\overbrace{\hspace{4em}}^{\bar\theta - i\delta}`}</div>
        </ForeignObjectLabel>
        <ForeignObjectLabel cx={bartheta + 2.5 * delta2} cy={15} width={300} height={300}>
            <div style={{ fontSize: "0.7em" }}>{Tex`\overbrace{\hspace{4em}}^{\bar\theta + j\delta}`}</div>
        </ForeignObjectLabel>
    </>
    return <Diagram minimize={minimize} predictedPath={predictedPath} hideBartheta />
}


const Diagram = ({ predictedPath, minimize, hideBartheta = false }) => {
    return <svg style={{
        width: `${width}px`, height: `${height}px`,
        transition: 'transform 0.2s',
        transformOrigin: `center 25px`,
        transform: minimize ? 'scale(0.6)' : 'scale(1.2)'
    }}>
        <g strokeWidth={1.5} fill="none">
            <line stroke="lightgray" x1={bartheta} y1={padding} x2={bartheta} y2={height - padding} />
            {!hideBartheta ? <ForeignObjectLabel cx={bartheta} cy={10}>
                <span style={{ fontSize: '0.7em', transition: 'opacity 0.2s', opacity: minimize ? 1 : 1 }}>
                    {Tex`\bar\theta`}
                </span>
            </ForeignObjectLabel> : null}
            <Axes origin={origin} width={width} padding={padding} />
            {predictedPath}
            <path d={d.M(origin[0], origin[1]) + d.l(50, -50) + d.l(10, 0) + d.l(5, 2.5)} stroke='black' />
        </g>
    </svg>
}


export const Axes = ({ origin, width, padding }) => {
    const arrow = 5
    return <>
        <line x1={origin[0]} y1={origin[1]} x2={origin[0]} y2={padding} stroke='gray' />
        <line x1={origin[0]} y1={origin[1]} x2={width - padding} y2={origin[1]} stroke='gray' />
        <path d={d.M(origin[0] - arrow, padding + arrow) + d.l(arrow, -arrow) + d.l(arrow, arrow)} stroke='gray' />
        <path d={d.M(width - padding - arrow, origin[1] - arrow) + d.l(arrow, arrow) + d.l(-arrow, arrow)} stroke='gray' />
        <ForeignObjectLabel cx={width - padding + 10} cy={origin[1]}>
            <span style={{ fontSize: '0.8em' }}>{Tex`\theta`}</span>
        </ForeignObjectLabel>
    </>
}
