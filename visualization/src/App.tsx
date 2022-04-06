import { Button, Navbar, NavbarGroup, NavbarHeading } from "@blueprintjs/core"
import * as React from "react"
import { ReactNode, useRef } from "react"
import styled from 'styled-components'
import useSize from '@react-hook/size'

const MyContainer = styled.div`
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
`;

const SvgContainer = (props: {
    children: ((props: {width: number, height:number}) => ReactNode)
}) => {
    const divRef = useRef(null);
    const [width, height] = useSize(divRef);
    return <div style={{width: "100%", height: "100%"}} ref={divRef}>
        { props.children({ width, height }) }
    </div>;
};

export default () => {
    const onClick = () => {
        
    }

    return <MyContainer>
        <Navbar><NavbarGroup><NavbarHeading>Dynamic Flow Visualization</NavbarHeading></NavbarGroup></Navbar>
        <Button intent="primary" onClick={onClick}>Load Network</Button>
        <div style={{flex: 1, position: "relative", overflow: "hidden"}}>
            <SvgContainer>{({width, height}) => <svg width={width} height={height} style={{position: "absolute", top:"0", left: "0"}}>
                <line x1={0} x2={width} y1={0} y2={height} stroke="black" />
            </svg>}</SvgContainer>
        </div>
    </MyContainer>
}; 