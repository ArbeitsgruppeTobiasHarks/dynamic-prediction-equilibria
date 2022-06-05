import { Flow } from "./Flow";

import sampleData from "./sampleFlowData.json"
import { Network } from './Network';


export const network = Network.fromJson(sampleData["network"])
export const flow = Flow.fromJson(sampleData["flow"])

