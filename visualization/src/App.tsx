import { Alignment, Button, ButtonGroup, Icon, Navbar, NavbarGroup, NavbarHeading, Slider } from "@blueprintjs/core"
import * as React from "react"
import { ReactNode, useRef } from "react"
import TeX from '@matejmazur/react-katex'

import styled from 'styled-components'
import useSize from '@react-hook/size'
import { flow, network } from "./example3"
import { Network } from "./Network"
import { Flow } from "./Flow"
import * as _ from "lodash"
import { max, min } from "lodash"
import { calcOutflowSteps, FlowEdge, SvgDefs, Vertex } from "./DynFlowSvg"

const MyContainer = styled.div`
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
`;

const useMinMax = (flow: Flow) => React.useMemo(
    () => {
        const allTimes = _.concat(
            _.map(flow.inflow.flat(), pwc => pwc.times).flat(),
            _.map(flow.outflow.flat(), pwc => pwc.times).flat(),
            _.map(flow.queues, pwc => pwc.times).flat()
        ) 

        return [min(allTimes), max(allTimes)]
    }
, [flow])

const DynamicFlowViewer = (props: { network: Network, flow: Flow }) => {
    const [t, setT] = React.useState(0)
    const [minT, maxT] = useMinMax(flow)
    const svgContainerRef = useRef(null);
    const [width, height] = useSize(svgContainerRef);
    return <>
        <div style={{ display: 'flex', padding: '16px', alignItems: 'center', overflow: 'hidden' }}>
            <Icon icon={'time'} /><div style={{ paddingLeft: '8px', paddingRight: '16px'}}>Time:</div> 
            <Slider onChange={(value) => setT(value)} value={t} min={minT} max={maxT} labelStepSize={(maxT - minT) / 10} />
        </div>
        <div style={{flex: 1, position: "relative", overflow: "hidden"}} ref={svgContainerRef}>
            <svg width={width} height={height} style={{position: "absolute", top:"0", left: "0"}}>
                <SvgContent  t={t} network={network} flow={flow} />
            </svg>
        </div>
    </>
}


export const SvgContent = ({t = 0, network, flow} : {t: number, network: Network, flow: Flow}) => {
    const svgIdPrefix = ""
    const outflowSteps = React.useMemo(
        () => flow.outflow.map((outflow: any) => calcOutflowSteps(outflow, ['#a00', '#0a0'])),
        [flow]
    )
    return <>
        <SvgDefs svgIdPrefix={svgIdPrefix} />
        {
            _.map(network.edgesMap, (value, id) => {
                const fromNode = network.nodesMap[value.from]
                const toNode = network.nodesMap[value.to]
                return <FlowEdge
                    key={id} t={t} capacity={value.capacity} from={[fromNode.x, fromNode.y]} to={[toNode.x, toNode.y]} svgIdPrefix={svgIdPrefix}
                    outflowSteps={outflowSteps[id]} transitTime={value.transitTime} queue={flow.queues[id]}
                />;
            })
        }
        {
            _.map(network.nodesMap, (value, id) => {
                return <Vertex key={id} pos={[value.x, value.y]} label={<TeX>{value.label ?? value.id}</TeX>} />
            })
        }
    </>
}



export default () => {
    return <MyContainer>
        <Navbar>
            <NavbarGroup>
                <NavbarHeading>Dynamic Flow Visualization</NavbarHeading>
            </NavbarGroup>
            <NavbarGroup align={Alignment.RIGHT}>
                <ButtonGroup><Button intent="primary" icon="folder-shared">Open Dynamic Flow</Button></ButtonGroup>
            </NavbarGroup>
        </Navbar>
        <DynamicFlowViewer network={network} flow={flow} />
    </MyContainer>
}; 