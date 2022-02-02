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
import { ConstantPredictorSvg, LinearPredictorSvg, RegressionPredictorSvg, RegularizedLinearPredictorSvg, ZeroPredictorSvg } from './predictorFigures';
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
        secondary: '#455a64',
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

const SubHeading = (props) => <Text color="secondary" textAlign="center" fontSize="h3" {...props} />

const TITLE = "Machine-Learned Prediction Equilibrium for Dynamic Traffic Assignment"

const CustomSlide = () => null

const CustomBox = (props) => <Box margin={20} borderStyle='1px solid lightgray' borderRadius={20} backgroundColor={theme.colors.tertiary} {...props} />

const Poster = () => (
    <Deck theme={theme}>
        <Slide backgroundColor='white' textColor={theme.colors.primary}>
            <Heading color={theme.colors.secondary} style={{ margin: '0px', padding: '0px' }}>{TITLE}</Heading>
            <Text className="authors" textAlign="center" fontSize="h2" style={{ margin: '0.5em', padding: '0px' }}>Lukas Graf<sup>1</sup>, Tobias Harks<sup>1</sup>, Kostas Kollias<sup>2</sup>, and Michael Markl<sup>1</sup>
                <div style={{ fontSize: "0.8em", margin: "0.625em", display: "flex", justifyContent: "center" }}><span style={{ width: "300px" }}><b>1</b>: University of Augsburg</span><span style={{ width: "300px" }}><b>2</b>: Google</span></div>
            </Text>
            <div style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-evenly' }}>
                <CustomBox width={1075}>
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
                                    <ShowcaseFormula text="Flow is conserved:" formula={
                                        BTex`\sum_{e\in\delta_v^+} f^+_{i,e}(\theta) - \sum_{e\in\delta_v^-} f^-_{i,e}(\theta) 
              \begin{cases}
              = u_i(\theta), & \text{if $v = s_i$}, \\
              = 0, & \text{if $v \notin \{s_i, t_i \}$}, \\
              \leq 0, & \text{if $v = t_i$}.
              \end{cases}`
                                    } />
                                    <ShowcaseFormula text="Queues operate at capacity:" formula={BTex`f_e^-(\theta) = \begin{cases}
            \nu_e,&\text{if $q_e(\theta - \tau_e) > 0$,} \\
            \min\{ f_e^+(\theta- \tau_e), \nu_e \}, &\text{otherwise.}
          \end{cases}`} />
                                    <ShowcaseFormula text="Capacity is split fairly:" formula={BTex`
                f_{i,e}^-(\theta) = f_e^-(\theta) \cdot \frac{f_{i,e}^+(\xi)}{f_e^+(\xi)}
                \quad\text{for $\xi\coloneqq \min\{\xi\leq\theta \mid \xi + \tau_e + \frac{q_e(\xi)}{\nu_e} = \theta \}$ with $f_e^+(\xi) > 0$}`} />
                                </UnorderedList>
                            </Definition>

                        </div>
                    </Box>
                </CustomBox>
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
                <CustomBox width={1075}>
                    <SubHeading>Sufficient Conditions for the Existence of DPE</SubHeading>
                    <Definition>A predictor {Tex`\hat q_{i,e}`} is <i>continuous</i>, if {BTex`
        \hat q_{i,e} : \mathbb R_{\geq0} \times \mathbb R_{\geq 0} \times C(\mathbb R_{\geq0}, \mathbb R_{\geq0})^{E} \to \mathbb R_{\geq 0},
        `} is continuous from the product topology,
                        where all {Tex` C(\mathbb R_{\geq0}, \mathbb R_{\geq0})`} are equipped with the topology induced by the uniform norm,
                        to {Tex`\R_{\geq 0}`}.
                    </Definition>
                    <Definition>
                        A predictor {Tex`\hat q_{i,e}`} is <i>oblivious</i>, if for all {Tex`\bar\theta \in\mathbb R_{\geq0}`} it holds {Tex`
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
                <CustomBox width={1075}>
                    <SubHeading>Applied Predictors</SubHeading>
                    <div style={{
                        marginLeft: "900px", width: "200px", textAlign: "center",
                        fontFamily: "'Open Sans'", fontSize: theme.fontSizes.text
                    }}>Compatible with Existence-Theorem</div>
                    <UnorderedList>
                        <PredictorListItem text={<>
                            <i>The Zero-Predictor </i>{Tex`\hat q^{\text{Z}}_{i,e}(\theta;\bar\theta;q) \coloneqq 0`}.<br />
                            <p>Predicted shortest paths always remain the same.</p>
                        </>} figure={(minimize) => <ZeroPredictorSvg minimize={minimize} />} compatible />
                        <PredictorListItem text={<>
                            <i>The constant predictor </i>{Tex`\hat q^{\text{C}}_{i,e}(\theta;\bar\theta;q) \coloneqq q_e(\bar\theta)`}.<br />
                            <p>Assumes the current conditions for the future.</p>
                        </>} figure={minimize => <ConstantPredictorSvg minimize={minimize} />} compatible />
                        <PredictorListItem text={<>
                            <i>The linear predictor </i>{Tex`\hat q^{\text{L}}_{i,e}(\theta;\bar\theta;q) \coloneqq 
          \left( q_e(\bar \theta)+\partial_-q_e(\bar \theta)\cdot \min\{ \theta-\bar\theta, H \} \right)^+
          `}.
                            <p>Not continuous in {Tex`\bar\theta`} whenever {Tex`\partial_-q_e`} jumps.</p>
                        </>} figure={minimize => <LinearPredictorSvg minimize={minimize} />} compatible={false} />
                        <PredictorListItem text={<>
                            <i>The regularized linear predictor </i><br />
                            <div style={{ textAlign: 'center' }}>{Tex`\hat q_{i,e}^{\text{RL}}(\theta;\bar\theta; q) \coloneqq
\Big( q_e(\bar\theta) + \frac{q_e(\bar\theta) - q_e(\bar\theta - \delta)}{\delta} \cdot \min\{ \theta - \bar\theta, H \} \Big)^+
      .`}</div>
                        </>} figure={(minimize) => <RegularizedLinearPredictorSvg minimize={minimize} />} compatible />
                        <PredictorListItem text={<>
                            <i>The linear regression predictor </i> {Tex`\hat q_{i,e}^{\text{ML}}`} linearly interpolates the points <br />
                            <div style={{ textAlign: 'center' }}>
                                <MLPredictorStepper />
                            </div>
                        </>} figure={(minimize) => <RegressionPredictorSvg minimize={minimize} />} compatible />
                    </UnorderedList>
                </CustomBox>

            </div>
        </Slide>



        <CustomSlide section="III. Applied Predictors">
            <SubHeading textAlign="left">A generalization of popular models</SubHeading>
            <Text style={{ marginBottom: 0 }}>We are given a dynamic prediction equilibrium {Tex`(\hat q, f)`}.</Text>
            <Text style={{ marginBottom: 0, marginTop: 0 }}>If all commodites use</Text>
            <UnorderedList style={{ marginTop: 0 }}>
                <ListItem style={{ marginTop: 16 }}>the <i>constant predictor</i> {Tex`\hat q_{i,e}(\theta;\bar\theta;q)\coloneqq q_e(\bar\theta)`}, then {Tex`f`} is an <i>instantaneous dynamic equilibrium (IDE)</i>.</ListItem>
                <ListItem style={{ marginTop: 32 }}>the <i>perfect predictor</i> {Tex`\hat q_{i,e}(\theta;\bar\theta;q)\coloneqq q_e(\theta)`}, then {Tex`f`} is a <i>dynamic (Nash) equilibrium (DE)</i>.</ListItem>
            </UnorderedList>
            <Text>IDE and especially DE have been studied quite extensively in the past.</Text>
            <Text style={{ marginTop: 0 }}>DPE generalize both concepts with a more realistic scenario.</Text>
        </CustomSlide>

        <CustomSlide intro section="IV. Computational Study">
            <SubHeading textAlign="left">Extension-based Simulation</SubHeading>
            <UnorderedList>
                <ListItem>Approximate a DPE by rerouting agents in discrete time intervals {Tex`\bar\theta_k = k\cdot \varepsilon`}.</ListItem>
                <ListItem>We assume that the network inflow rates are piecewise constant with finite jumps</ListItem>
                <ListItem>The extension procedure for one routing interval {Tex`(\bar\theta_k,\bar\theta_{k+1})`} given an equilibrium flow up to time {Tex`H = \bar\theta_k`}:
                    <div style={{ width: '1200px' }}>
                        <ThemeProvider theme={{ size: { width: '1200px' } }}>
                            <OrderedList style={{ backgroundColor: 'white', border: '1px solid lightgray', fontFamily: '' }}>
                                <ListItem>Gather predictions {Tex`(\hat q_{i,e}(\,\boldsymbol\cdot\,;\bar\theta_k; q))_{i,e}`} for {Tex`\bar\theta_k`}</ListItem>
                                <ListItem>Compute all shortest {Tex`v`}-{Tex`t_i`}-paths at time {Tex`\bar\theta_k`} predicted at time {Tex`\bar\theta_k`}</ListItem>
                                <ListItem><Code>while </Code>{Tex`H < \bar\theta_{k+1}`}<Code> do:</Code></ListItem>
                                <ListItem><Code>    </Code>Compute maximal {Tex`H'\leq\bar\theta_{k+1}`} such that {Tex`b_{i,v}^-(\theta)\coloneqq \sum_{e\in\delta_{v}^-} f_{i,e}^-(\theta) + u_i(\theta)\cdot\mathbf{1}_{v=s_i}`} is constant on {Tex`(H, H')`} for all {Tex`v\in V, i\in I`}</ListItem>
                                <ListItem><Code>    </Code>Equally distribute {Tex`b_{i,v}^-(\theta)`} to the outgoing edges lying on shortest paths during {Tex`(H, H')`}</ListItem>
                                <ListItem><Code>    </Code>{Tex`H \leftarrow H'`}</ListItem>
                            </OrderedList>
                        </ThemeProvider>
                    </div>
                </ListItem>
            </UnorderedList>
        </CustomSlide>

        <CustomSlide section="IV. Computational Study">
            <SubHeading textAlign="left">Comparing the Performance of Predictors</SubHeading>
            <UnorderedList>
                <ListItem>
                    We monitor the average travel time of particles over multiple DPE simulations with varying inflow rates.
                </ListItem>
                <ListItem>
                    For a sample network, the linear regression already performs best:
                    <Minimizer>{minimize =>
                        <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-evenly', marginTop: '20px' }}>
                            <div style={{
                                transition: 'transform 0.2s', transform: minimize ? 'translateY(0)' : 'translateY(80px) scale(1.2)',
                                textAlign: "center"
                            }}>
                                <img src={sampleNetwork} width='200px' />
                                <Text style={{ margin: 0 }}>Edges are labeled with {Tex`(\tau_e, \nu_e)`}</Text>
                            </div>
                            <div>
                                <img style={{
                                    transition: 'transform 0.2s', transform: minimize ? 'scale(1)' : 'scale(1.8)',
                                    transformOrigin: 'top', width: "280px"
                                }} src={performance} />
                            </div>
                        </div>}
                    </Minimizer>
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
        </CustomSlide>

        <CustomSlide intro section="V. Conclusion">
            <CustomTable style={{ margin: "100px auto", textAlign: "center" }} width={0.8}>
                <TableHeader textAlign="center">
                    <th>Contributions</th>
                    <th>Future Work</th>
                </TableHeader>
                <colgroup>
                    <col style={{ width: '50%' }} />
                    <col style={{ width: '50%' }} />
                </colgroup>
                <tr>
                    <td>
                        <UnorderedList style={{ display: "inline-block" }}>
                            <ListItem>We formulated a mathematically concise model that generalizes existing rather unrealistic models.</ListItem>
                            <ListItem>In this model, we proved the existence of equilibria under mild assumptions on the predictors.</ListItem>
                            <ListItem>The framework allows the integration of arbitrary ML methods as predictors.</ListItem>
                        </UnorderedList>
                    </td>
                    <td>
                        <UnorderedList style={{ display: "inline-block" }}>
                            <ListItem>Generalize the predictor's input to allow for other flow related data than past queues.</ListItem>
                            <ListItem>Embed more advanced ML methods for traffic forecast into the simulation.</ListItem>
                        </UnorderedList>
                    </td>
                </tr>

            </CustomTable>
        </CustomSlide>
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
    return Tex`
          \left(
            \bar\theta + j\delta,
             \left(  
              \sum_{e' \in N(e)} 
                 \sum_{i=0}^k a_{i,j}^{e'}\cdot q_{e'}(\bar\theta-i\delta)
              \right)^+
          \right)
      .`
}

const PredictorListItem = ({ text, figure, compatible }) => {
    return <ListItem>
        <div style={{ display: 'flex', flexDirection: 'row' }}>
            <div style={{ width: '650px', height: '100px' }}>{text}</div>
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
`

ReactDOM.render(<AllowOverflow><Poster /></AllowOverflow>, document.getElementById('root'))
