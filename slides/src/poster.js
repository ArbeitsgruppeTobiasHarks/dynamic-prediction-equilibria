import React, { useRef } from 'react';
import ReactDOM from 'react-dom';
import {
    FlexBox,
    Heading,
    UnorderedList,
    ListItem as OriginalListItem,
    Stepper,
    Slide,
    Deck,
    Text,
    Box,
    Appear,
    OrderedList,
    Table,
    TableHeader
} from 'spectacle';
import styled from 'styled-components'
import performance from './performance2.png'
import sampleNetwork from './network2.png'
import { ThemeProvider } from 'styled-components'
import syntaxTheme from 'react-syntax-highlighter/dist/cjs/styles/prism/vs';

import { animated, useSpring, useChain } from 'react-spring';
import { Example1Svg } from './example1';
import { Example2Svg } from './example2';
import { BTex, Tex } from './tex';
import { ConstantPredictorSvg, LinearPredictorSvg, PerfectPredictorSvg, RegressionPredictorSvg, RegularizedLinearPredictorSvg, ZeroPredictorSvg } from './predictorFigures';
import { Example3Svg } from './example3';

const ListItem = (props) => <OriginalListItem style={{ margin: "10px" }}  {...props} />


const theme = {
    fonts: {
        header: '"Open Sans", Helvetica, Arial, sans-serif',
        text: '"Open Sans", Helvetica, Arial, sans-serif',
        head: '"Open Sans", Helvetica, Arial, sans-serif'
    },
    fontWeights: {
        bold: 600
    },
    colors: { // https://material.io/resources/color/#!/?view.left=0&view.right=1&primary.color=B0BEC5&secondary.color=F57F17
        primary: '#000',
        black: '#000',
        secondary: '#41738b', // '#455a64',
        tertiary: '#f5f5f5',
        background: 'white'
    },
    fontSizes: {
        h1: '48px',
        h2: '32px',
        h3: '28px',
        head: '16px',
        text: '18px',
    },
    size: {
        width: 3840,
        height: 2160
    }
}

const SubHeading = (props) => <Text color="secondary" textAlign="center" fontWeight='700' fontSize='32px' {...props} />

const TITLE = "Machine-Learned Prediction Equilibrium for Dynamic Traffic Assignment"

const CustomBox = (props) => <Box margin={40} borderStyle='1px solid lightgray' borderRadius={20} backgroundColor={theme.colors.tertiary} {...props} />


const PhysicalFlowModelBox = () => (
    <CustomBox width={1075} position="relative" >
        <SubHeading color={theme.colors.secondary}>The Physical Flow Model</SubHeading>
        <Box color={theme.colors.primary}>
            <div>
                <Text style={{ margin: "0 32px", padding: "0" }}>We are given</Text>
                <UnorderedList style={{ margin: "0 32px" }}>
                    <ListItem>a finite, directed graph {Tex`G=(V,E)`},</ListItem>
                    <ListItem>edge transit times {Tex`\tau_e > 0`} and edge capacities {Tex`\nu_e> 0`} for {Tex`e\in E`}, and</ListItem>
                    <ListItem>commodities {Tex`i\in I`} each with source and sink nodes {Tex`s_i, t_i\in V`} and <br />a network inflow rate {Tex`u_i: \mathbb R_{\geq 0} \to \mathbb R_{\geq 0}`}.</ListItem>
                </UnorderedList>
                <Definition>
                    A <i>dynamic flow</i> {Tex`f=(f^+, f^-)`} consists of
                    <UnorderedList style={{ margin: "0" }}>
                        <ListItem>edge inflow rates {Tex`f^+_{i,e}:\mathbb R_{\geq 0}\to \mathbb R_{\geq 0}`} for {Tex`i\in I, e\in E`} and</ListItem>
                        <ListItem>edge outflow rates {Tex`f^-_{i,e}: \mathbb R_{\geq 0}\to \mathbb R_{\geq 0}`} for {Tex`i\in I, e\in E`}.</ListItem>
                    </UnorderedList>
                </Definition>
                <Notation>
                    {Tex`f_e^+ \coloneqq \sum_{i\in I} f_{i,e}^+,`}
                    {Tex`~~f_e^- \coloneqq \sum_{i\in I} f_{i,e}^-,`}
                    {Tex`~~q_e(\theta) \coloneqq \int_0^\theta f^+_e(z) - f^-_e(z+\tau_e) \,\mathrm dz`}
                </Notation>
                <Definition>
                    A dynamic flow {Tex`f`} is <i>feasible</i> if it fulfills the following conditions:
                    <UnorderedList style={{ margin: "0" }}>
                        <ShowcaseFormula text="Flow is conserved:" formula={BTex`\sum_{e\in\delta_v^+} f^+_{i,e}(\theta) - \sum_{e\in\delta_v^-} f^-_{i,e}(\theta) 
              \begin{cases}
              = u_i(\theta), & \text{if $v = s_i$}, \\
              = 0, & \text{if $v \notin \{s_i, t_i \}$}, \\
              \leq 0, & \text{if $v = t_i$}.
              \end{cases}`} />
                        <ShowcaseFormula text="Queues operate at capacity:" formula={BTex`f_e^-(\theta) = \begin{cases}
            \nu_e,&\text{if $q_e(\theta - \tau_e) > 0$,} \\
            \min\{ f_e^+(\theta- \tau_e), \nu_e \}, &\text{otherwise.}
          \end{cases}`} />
                        <ShowcaseFormula text="Capacity is split fairly:" formula={BTex`
                f_{i,e}^-(\theta) = f_e^-(\theta) \cdot \frac{f_{i,e}^+(\xi)}{f_e^+(\xi)}
                \quad\text{for $\xi\coloneqq \min\{\xi\leq\theta \mid \xi + \tau_e + \frac{q_e(\xi)}{\nu_e} = \theta \}$ with $f_e^+(\xi) > 0$}.`} />
                    </UnorderedList>
                </Definition>
            </div>
        </Box>
        <div style={{ position: 'absolute', right: '16px ', top: '100px', width: '337.5px' }}>
            <div style={{ transformOrigin: 'left top', transform: 'scale(0.75)', height: 250 }}>
                <Example1Svg overrideT={1450} />
            </div>
            <Text style={{ textAlign: 'center', margin: '0', padding: '0' }}>The queueing model.</Text>
        </div>
    </CustomBox>
)

const BehavioralModelBox = () => (
    <CustomBox width={1075}>
        <SubHeading>The Behavioral Model</SubHeading>
        <Box>
            <UnorderedList margin="0 32px">
                <ListItem>The <i>exit time</i> when entering edge {Tex`e`} at time {Tex`\theta`} is given by {Tex`T_e(\theta)\coloneqq \theta + \tau_e + \frac{q_e(\theta)}{\nu_e}`}</ListItem>
                <ListItem>Each commodity {Tex`i\in I`} is equipped with a set of <i>predictors</i> {BTex`
                    \hat q_{i,e} : \mathbb R_{\geq0} \times \mathbb R_{\geq 0} \times C(\mathbb R_{\geq0}, \mathbb R_{\geq0})^{E} \to \mathbb R_{\geq 0},
                    \quad
                    (\theta, \bar\theta, q)\mapsto\hat q_{i,e}(\theta; \bar\theta; q),`}
                    where {Tex`\hat q_{i,e}(\theta; \bar\theta; q)`} describes the <i>predicted queue length </i>
                    of edge {Tex`e`} at time {Tex`\theta`} as predicted at time {Tex`\bar\theta`} using the historical queue functions {Tex`q`}.</ListItem>
                <ListItem>The <i>predicted exit time</i> when entering an edge {Tex`e`} at time {Tex`\theta`} is given by {Tex`\hat T_{i,e}(\theta; \bar\theta; q)\coloneqq \theta + \tau_e + \frac{\hat q_{i,e}(\theta; \bar\theta, q)}{\nu_e}`}.</ListItem>
                <ListItem>The <i>predicted exit time</i> when entering a path {Tex`P=(e_1, \dots, e_k)`} at time {Tex`\theta`} is given by
                    {BTex`\hat T_{i,P}(\theta; \bar\theta; q)
                    \coloneqq \left(\hat T_{e_k}(\,\boldsymbol{\cdot}\,;\bar\theta;q) \circ \cdots \circ \hat T_{e_1}(\,\boldsymbol{\cdot}\,;\bar\theta;q)\right)(\theta).
                    `}
                </ListItem>
                <ListItem>
                    The <i>predicted earliest arrival</i> at {Tex`t_i`} when starting at time {Tex`\theta`} at {Tex`v`} is given by
                    {BTex`\hat l_{i,v}(\theta; \bar\theta; q)
                    \coloneqq \min_{P\text{ simple } v\text{-}t_i\text{-path}} \hat T_{i,P}(\theta;\bar\theta;q).
                    `}
                </ListItem>
            </UnorderedList>
            <Definition>
                A pair {Tex`(\hat q, f)`} of predictors {Tex`\hat q = (\hat q_{i,e})_{i\in I, e\in E}`} and
                a feasible dynamic flow {Tex`f`} is a <i>dynamic prediction equilibrium (DPE)</i>, if for all edges {Tex`e=vw`} and all {Tex`\theta \geq 0`} it holds that
                {BTex`
                    f^+_{i,e}(\theta) > 0 \implies \hat l_{i,v}(\theta;\theta; q) = \hat l_{i,w}(\hat T_{i,e}( \theta;\theta; q ); \theta; q).
                    `}
            </Definition>
        </Box>
    </CustomBox>
)


const ExistenceBox = () => (
    <CustomBox width={1075}>
        <SubHeading>Sufficient Conditions for the Existence of DPE</SubHeading>
        <Definition>A predictor {Tex`\hat q_{i,e}`} is <i>continuous</i>, if {BTex`
            \hat q_{i,e} : \mathbb R_{\geq0} \times \mathbb R_{\geq 0} \times C(\mathbb R_{\geq0}, \mathbb R_{\geq0})^{E} \to \mathbb R_{\geq 0},
            `} is continuous from the product topology,
            where all {Tex` C(\mathbb R_{\geq0}, \mathbb R_{\geq0})`} are equipped with the topology induced by the uniform norm,
            to {Tex`\R_{\geq 0}`}.
        </Definition>
        <Definition>
            A predictor {Tex`\hat q_{i,e}`} is <i>oblivious</i>, if for all {Tex`\bar\theta \in\mathbb R_{\geq0}`} it holds {BTex`
            \quad\forall q,q'\colon\quad
            q_{\hspace{.07em}\vert\hspace{.07em}[0, \bar\theta]^E} = q'_{\hspace{.07em}\vert\hspace{.07em}[0, \bar\theta]^E}
            \implies
            \hat q_{i,e}(\,\boldsymbol{\cdot}\,;\bar\theta;q)=\hat q_{i,e}(\,\boldsymbol{\cdot}\,;\bar\theta;q').
            `}
        </Definition>

        <Definition>
            A predictor {Tex`\hat q_{i,e}`} <i>respects FIFO</i>, if {Tex`\hat T_{i,e}(\,\boldsymbol{\cdot}\,;\bar\theta, q)`} is non-decreasing
            for all {Tex`\bar\theta\in \R_{\geq0}`} and {Tex` q\in C(\mathbb R_{\geq0},\mathbb R_{\geq0})^{E}`}.
        </Definition>

        <Theorem>
            If all network inflow rates {Tex`u_i`} are bounded and all predictors {Tex`\hat q_{i, e}`} are
            continuous, oblivious, and respect FIFO, then
            there exists a dynamic prediction equilibrium {Tex`(\hat q, f)`}.
        </Theorem>

    </CustomBox>
)

const AnalyzedPredictorsBox = () => (
    <CustomBox width={1075}>
        <SubHeading>The Analyzed Predictors</SubHeading>
        <div style={{
            marginLeft: "875px", width: "200px", textAlign: "center",
            fontFamily: "'Open Sans'", fontSize: theme.fontSizes.text
        }}>Compatible with Existence-Theorem</div>
        <UnorderedList>
            <PredictorListItem text={<>
                <i>The Zero-Predictor </i>{Tex`\hat q^{\text{Z}}_{i,e}(\theta;\bar\theta;q) \coloneqq 0`}.<br />
                <p style={{ marginTop: '5px' }}>Predicted shortest paths always remain the same.</p>
            </>} figure={(minimize) => <ZeroPredictorSvg minimize={minimize} />} compatible />
            <PredictorListItem text={<>
                <i>The constant predictor </i>{Tex`\hat q^{\text{C}}_{i,e}(\theta;\bar\theta;q) \coloneqq q_e(\bar\theta)`}.
                <p style={{ marginTop: '5px' }}>Assumes the current conditions for the future. If all commodities use this predictor, a DPE corresponds to an Instantaneous Dynamic Equilibrium.</p>
            </>} figure={minimize => <ConstantPredictorSvg minimize={minimize} />} compatible />
            <PredictorListItem text={<>
                <i>The linear predictor </i>{Tex`\hat q^{\text{L}}_{i,e}(\theta;\bar\theta;q) \coloneqq 
                \left( q_e(\bar \theta)+\partial_-q_e(\bar \theta)\cdot \min\{ \theta-\bar\theta, H \} \right)^+
                `}.
                <p style={{ marginTop: '5px' }}>Not continuous in {Tex`\bar\theta`} whenever {Tex`\partial_-q_e`} jumps.</p>
            </>} figure={minimize => <LinearPredictorSvg minimize={minimize} />} compatible={false} />
            <PredictorListItem text={<>
                <i>The regularized linear predictor </i><br />
                <div style={{ textAlign: 'center' }}>{Tex`\hat q_{i,e}^{\text{RL}}(\theta;\bar\theta; q) \coloneqq
        \Big( q_e(\bar\theta) + \frac{q_e(\bar\theta) - q_e(\bar\theta - \delta)}{\delta} \cdot \min\{ \theta - \bar\theta, H \} \Big)^+
       .`}</div>
            </>} figure={(minimize) => <RegularizedLinearPredictorSvg minimize={minimize} />} compatible />
            <PredictorListItem text={<>
                <i>The linear regression predictor </i> {Tex`\hat q_{i,e}^{\text{ML}}`} linearly interpolates the points <br />
                <div style={{ textAlign: 'center' }}>{Tex`
            \left(
                \bar\theta + j\delta,
                \left(  
                \sum_{e' \in N(e)} 
                    \sum_{i=0}^k a_{i,j}^{e'}\cdot q_{e'}(\bar\theta-i\delta)
                \right)^+
            \right)
        .`}</div>
            </>} figure={(minimize) => <RegressionPredictorSvg minimize={minimize} />} compatible />
            <PredictorListItem text={<>
                <i>The perfect predictor </i> {Tex`\hat q^{\text{P}}_{i,e}(\theta;\bar\theta;q) \coloneqq q_e(\theta)`}.
                <p style={{ marginTop: '5px' }}>Will always predict the future correctly and is thus not oblivious. If all commodities use this predictor, a DPE corresponds to a dynamic equilibrium in the full-information model.</p>
            </>} figure={(minimize) => <PerfectPredictorSvg minimize={minimize} />} compatible={false} />
        </UnorderedList>
    </CustomBox>
)

const SimulationBox = () => (
    <CustomBox width={1075}>
        <SubHeading>Extension-based Simulation</SubHeading>
        <UnorderedList>
            <ListItem>Approximate a DPE by rerouting agents in discrete time intervals {Tex`\bar\theta_k = k\cdot \varepsilon`}.</ListItem>
            <ListItem>We assume that the network inflow rates are piecewise constant with finite jumps</ListItem>
            <ListItem>The extension procedure for one routing interval {Tex`(\bar\theta_k,\bar\theta_{k+1})`} given an equilibrium flow up to time {Tex`H = \bar\theta_k`}:
                <div style={{ width: '900px' }}>
                    <ThemeProvider theme={{ size: { width: '900px' } }}>
                        <OrderedList style={{ backgroundColor: 'white', border: '1px solid lightgray', fontFamily: '' }}>
                            <ListItem>Gather predictions {Tex`(\hat q_{i,e}(\,\boldsymbol\cdot\,;\bar\theta_k; q))_{i,e}`} for {Tex`\bar\theta_k`}</ListItem>
                            <ListItem>Compute all shortest {Tex`v`}-{Tex`t_i`}-paths at time {Tex`\bar\theta_k`} predicted at time {Tex`\bar\theta_k`}</ListItem>
                            <ListItem><Code>while </Code>{Tex`H < \bar\theta_{k+1}`}<Code> do:</Code></ListItem>
                            <ListItem><Code>    </Code><div style={{ display: 'inline-block', verticalAlign: 'text-top' }}>Compute maximal {Tex`H'\leq\bar\theta_{k+1}`} such that {Tex`b_{i,v}^-(\theta)\coloneqq \sum_{e\in\delta_{v}^-} f_{i,e}^-(\theta) + u_i(\theta)\cdot\mathbf{1}_{v=s_i}`}<br /> is constant on {Tex`(H, H')`} for all {Tex`v\in V, i\in I`}</div></ListItem>
                            <ListItem><Code>    </Code>Equally distribute {Tex`b_{i,v}^-(\theta)`} to the outgoing edges lying on shortest paths during {Tex`(H, H')`}</ListItem>
                            <ListItem><Code>    </Code>{Tex`H \leftarrow H'`}</ListItem>
                        </OrderedList>
                    </ThemeProvider>
                </div>
            </ListItem>
            <ListItem>This simulation enables us to generate training data for the linear regression predictor {Tex`\hat q^{\text{ML}}_{i,e}`}:
            <UnorderedList>
                <ListItem>
                    We use the constant predictor to create sample DPEs.
                </ListItem>
                <ListItem>
                    This allows the model to estimate the progression of queues when agents follow our behavioral model.
                </ListItem>
                <ListItem>
                    In small networks, the weights have been trained seperately on each edge.
                </ListItem>
                <ListItem>
                    In larger networks, a single weight matrix has been learned using the collective data of all edges.
                </ListItem>
            </UnorderedList></ListItem>
        </UnorderedList>
    </CustomBox>
)

const ComparingPerformanceBox = () => (
    <CustomBox width={1075}>
        <SubHeading>Comparing the Performance of Predictors</SubHeading>
        <UnorderedList>
            <ListItem>
                We monitor the average travel time of particles over multiple DPE simulations with varying inflow rates.
            </ListItem>
            <ListItem>
                For a sample network, the linear regression already performs best:
                <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-evenly', marginTop: '20px' }}>
                    <div style={{
                        transform: 'translateY(100px)',
                        textAlign: "center"
                    }}>
                        <img src={sampleNetwork} width='200px' />
                        <Text style={{ margin: 0 }}>Edges are labeled with {Tex`(\tau_e, \nu_e)`}</Text>
                    </div>
                    <div>
                        <img style={{
                            transform: 'scale(1)',
                            transformOrigin: 'top', width: "500px"
                        }} src={performance} />
                    </div>
                </div>
            </ListItem>


            <ListItem>
                Simulations in real-world road traffic networks (centre of Tokyo, Sioux Falls) show
                <UnorderedList>
                    <ListItem>the linear regression predictor is amongst the best predictors analyzed,</ListItem>
                    <ListItem>the Zero-Predictor performs worst most of the time,</ListItem>
                    <ListItem>the simulation is capable of computing DPEs in large-scale networks.</ListItem>
                </UnorderedList>
            </ListItem>

        </UnorderedList>
    </CustomBox>
)

const TimeSeries = () => {
    const height = 300
    const width = 350
    return <div style={{ position: 'relative', height: 1975, left: '20px' }}>
        {
            [0, 100, 200, 300, 400, 500, 600, 700].map((t, ind) => {
                const top = -80 + ind * 200 + (ind >= 2 ? 10 : 0) + (ind >= 3 ? 25 : 0) + (ind >= 4 ? 55 : 0) + (ind >= 5 ? 95 : 0) + (ind >= 6 ? 55 : 0) + (ind >= 7 ? 25 : 0)
                return <div key={t} style={{ height, width, overflow: 'hidden', position: 'absolute', top }}>
                    <Example3Svg demo svgIdPrefix={`example3-t`} overrideT={t + 100} height={height} width={width} />
                </div>;
            })
        }
        <Text style={{ position: "absolute", bottom: 0, left: 0, right: 0, textAlign: "center" }}>A DPE using predictor {Tex`\hat q_{i,e}^L`}.</Text>
    </div>
}

// const pageSize = 38.5

const Poster = () => (
    <Deck theme={theme}>
        <Slide padding={0} backgroundColor='white' textColor={theme.colors.primary}>
            <Heading fontSize='64px' color={'white'} backgroundColor={theme.colors.secondary} style={{ margin: '-16px -16px 10px -16px', padding: '32px' }}>{TITLE}</Heading>
            <div style={{ display: 'flex', flexDirection: 'row' }}>
                <div style={{ width: '90%', height: 1975, display: 'flex', flexDirection: 'column' }}>
                    <Text className="authors" textAlign="center" fontSize="h2" style={{ margin: '0.5em', padding: '0px' }}>Lukas Graf<sup>1</sup>, Tobias Harks<sup>1</sup>, Kostas Kollias<sup>2</sup>, and Michael Markl<sup>1</sup>
                        <div style={{ fontSize: "0.8em", margin: "0.625em", display: "flex", justifyContent: "center" }}><span style={{ width: "300px" }}><b>1</b>: University of Augsburg</span><span style={{ width: "300px" }}><b>2</b>: Google</span></div>
                    </Text>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gridTemplateRows: '1fr 1fr', justifyContent: 'space-evenly', alignItems: 'stretch', flex: 1 }}>
                        <PhysicalFlowModelBox />
                        <BehavioralModelBox />
                        <ExistenceBox />
                        <AnalyzedPredictorsBox />
                        <SimulationBox />
                        <ComparingPerformanceBox />
                    </div>
                </div>
                <div style={{ width: '10%' }}>
                    <TimeSeries />
                </div>
            </div>
        </Slide>
    </Deck>
)

const CustomTable = styled(Table)`
  border-collapse: collapse;
& td {
  border: 2px solid ${theme.colors.secondary}; 
}
& tr:first-child td {
  border-top: 0;
}
& tr td, th {
  border-left: 0;
}
& tr:last-child td {
  border-bottom: 0;
}
& tr td, th {
  border-right: 0;
}

& li {
  padding: 20px;
}
`

const Code = (props) => <span style={{ whiteSpace: 'pre' }} {...props} />
const MLPredictorStepper = () => {
    return
}

const PredictorListItem = ({ text, figure, compatible }) => {
    return <ListItem>
        <div style={{ display: 'flex', flexDirection: 'row' }}>
            <div style={{ width: '625px', height: '100px' }}>{text}</div>
            <div style={{ height: '90px' }}>{figure(true)}</div>
            <div style={{ height: '100px', display: 'flex', justifyContent: 'center', alignItems: 'center', marginLeft: '60px' }}>
                {compatible ? '✔️' : '❌'}</div>
        </div>
    </ListItem>
}

const Minimizer = ({ children }) => {
    return <Stepper values={[true]} alwaysVisible>{
        (value, step, isActive) => {
            return children(value || false)
        }
    }
    </Stepper>
}

const LessTexMargin = styled('div')`
    & .katex-display {
        margin: 8px 0;
    }
`

const ShowcaseFormula = ({ formula, text }) => {
    return <ListItem style={{ margin: 0, padding: 0 }}>
        <LessTexMargin style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
            <div>{text}</div>
            <div style={{ padding: '0 10px' }}>{formula}</div>
        </LessTexMargin></ListItem>
}


const Notation = ({ children }) => {
    return <Box margin="32px" style={{ fontSize: theme.fontSizes.text, fontFamily: "Open Sans" }}>
        <span><i>Notation. </i></span>
        {children}
    </Box>
}

const Question = ({ children }) => {
    return <Box margin="32px" style={{ fontSize: theme.fontSizes.text, fontFamily: "Open Sans" }}>
        <span style={{ color: theme.colors.secondary }}><b>Question. </b></span>
        {children}
    </Box>
}

const Definition = ({ children }) => {
    return <Box margin="32px" style={{ fontSize: theme.fontSizes.text, fontFamily: "Open Sans" }}>
        <span style={{ color: theme.colors.secondary }}><b>Definition. </b></span>
        {children}
    </Box>
}

const Example = ({ children }) => {
    return <Box margin="32px" style={{ fontSize: theme.fontSizes.text, fontFamily: "Open Sans" }}>
        <span style={{ color: theme.colors.secondary }}><b>Example. </b></span>
        {children}
    </Box>
}

const Theorem = ({ children }) => {
    return <Box margin="32px" style={{ fontSize: theme.fontSizes.text, fontFamily: "Open Sans" }}>
        <span style={{ color: theme.colors.secondary }}><b>Theorem. </b></span>
        <i>{children}</i>
    </Box>
}

const AllowOverflow = styled.div`
    & > div {
        overflow: auto !important;
    }

    & > div > div {
        background: black;
    }
`

document.documentElement.style.background = "black"

ReactDOM.render(<AllowOverflow><Poster /></AllowOverflow>, document.getElementById('root'))
