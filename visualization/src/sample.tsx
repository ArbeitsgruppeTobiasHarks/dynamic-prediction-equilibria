import { Flow } from "./Flow";
import * as _ from 'lodash'

import sampleData from "./sampleFlowData.js"
import { Network } from './Network';


export const network = Network.fromJson(sampleData.network)
export const flow = Flow.fromJson(sampleData.flow)

