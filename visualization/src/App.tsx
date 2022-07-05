import { Alignment, Button, ButtonGroup, Navbar, NavbarGroup, NavbarHeading, Toaster } from "@blueprintjs/core"
import * as React from "react"
import { useRef, useState } from "react"

import styled from 'styled-components'
import { flow as initialFlow, network as initialNetwork } from "./sample"
import { Network } from "./Network"
import { Flow } from "./Flow"
import { DynamicFlowViewer } from "./DynamicFlowViewer"

const MyContainer = styled.div`
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;

    #drop {
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        bottom: 0;
        justify-content: center;
        align-items: center;
        display: none;
        z-index: 0;
    }

    &:drop #drop {
        display: flex;
    }
`;


export default () => {
    const [{ network, flow }, setNetworkAndFlow] = useState({ network: initialNetwork, flow: initialFlow })
    const [dragOver, setDragOver] = useState(false)
    const fileInputRef = useRef<HTMLInputElement>()


    const openFlowFromJsonText = (jsonText: string) => {
        try {
            const json = JSON.parse(jsonText)
            const network = Network.fromJson(json["network"])
            const flow = Flow.fromJson(json["flow"])
            setNetworkAndFlow({ network, flow })
            AppToaster.show({ message: "Dynamic Flow loaded.", intent: 'success' })
        } catch (error) {
            if (error instanceof SyntaxError) {
                console.error(error)
                AppToaster.show({ message: String(error), intent: "danger" })
            } else {
                console.error(error)
                AppToaster.show({ message: "Could not construct a flow: " + String(error), intent: "danger" })
            }
        }
    }

    const onOpen: React.FormEventHandler<HTMLInputElement> = (event: any) => {
        for (const file of event.target.files) {
            const reader = new FileReader()
            reader.addEventListener("load", () => {
                // @ts-ignore
                openFlowFromJsonText(reader.result)
            })
            reader.readAsText(file)
            reader.addEventListener("abort", () => AppToaster.show({ message: "Could not read file.", intent: "danger" }))
            reader.addEventListener("error", () => AppToaster.show({ message: "Could not read file.", intent: "danger" }))
        }
    }

    const onDrop = async (event: any) => {
        event.preventDefault()
        setDragOver(false)
        if (event.dataTransfer.items) {
            // Use DataTransferItemList interface to access the file(s)
            if (event.dataTransfer.items.length != 1) {
                AppToaster.show({ message: "Error. Please drop exactly one file.", intent: "danger" })
                return
            }
            const item = event.dataTransfer.items[0]
            // If dropped items aren't files, reject them
            if (item.kind !== 'file') {
                AppToaster.show({ message: "Please drop a file.", intent: 'danger' })
                return
            }
            const file: File = item.getAsFile()
            openFlowFromJsonText(await file.text())
        } else {
            // Use DataTransfer interface to access the file(s)
            if (event.dataTransfer.files.length != 1) {
                AppToaster.show({ message: "Please drop exactly one file.", intent: 'danger' })
                return
            }
            const file: File = event.dataTransfer.files[0]
            openFlowFromJsonText(await file.text())
        }
    }

    const onDragOver = (event: any) => {
        // Prevent default behavior (Prevent file from being opened)
        event.preventDefault();
        setDragOver(true)
    }

    return <MyContainer onDrop={onDrop} onDragOver={onDragOver} onDragLeave={() => setDragOver(false)}>
        <DragOverContainer dragOver={dragOver} />
        <Navbar>
            <NavbarGroup>
                <NavbarHeading>Dynamic Flow Visualization</NavbarHeading>
            </NavbarGroup>
            <NavbarGroup align={Alignment.RIGHT}>
                <ButtonGroup>
                    <Button icon="folder-shared" intent="primary" onClick={() => fileInputRef.current.click()}>Open Dynamic Flow</Button>
                    <input style={{ display: 'none' }} ref={fileInputRef} type="file" accept=".json" onChange={onOpen} />
                </ButtonGroup>
            </NavbarGroup>
        </Navbar>
        <DynamicFlowViewer network={network} flow={flow} />
    </MyContainer>
};

const DragOverContainer = ({ dragOver }: { dragOver: boolean }) => {
    return <div style={{
        position: 'absolute',
        top: '0px',
        bottom: '0px',
        left: '0px',
        right: '0px',
        display: 'flex',
        pointerEvents: 'none',
        zIndex: 2,
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '40px',
        opacity: dragOver ? '1' : '0',
        background: 'rgba(255,255,255,0.9)',
        transition: 'opacity 0.2s'
    }}>
        Drop your file.
    </div>
}

const AppToaster = Toaster.create()