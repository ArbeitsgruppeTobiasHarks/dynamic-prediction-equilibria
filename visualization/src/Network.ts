type IdType = number | string
type NodeId = IdType
type EdgeId = IdType

export class Network {
    edgesMap : { [id: EdgeId]: Edge }
    nodesMap: { [id : NodeId]: NetNode }

    constructor(nodes: NetNode[], edges: Edge[]) {
        this.nodesMap = nodes.reduce<{ [id: NodeId]: NetNode }>((acc, node) => {
            acc[node.id] = node
            return acc
        }, {})
        this.edgesMap = edges.reduce<{ [id: EdgeId]: Edge }>((acc, edge) => {
            acc[edge.id] = edge
            return acc
        }, {})
    }
}

export interface NetNode {
    id: NodeId
    label?: string
    x: number
    y: number
}

export interface Edge {
    id: EdgeId
    from: NodeId
    to: NodeId
    capacity: number
    transitTime: number    
}