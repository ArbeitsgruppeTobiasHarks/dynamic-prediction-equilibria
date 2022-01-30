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
    secondary: '#455a64',
    tertiary: '#f5f5f5'
  },
  fontSizes: {
    h1: '48px',
    h2: '32px',
    h3: '28px',
    head: '16px',
    text: '18px',
  }
}

const template = () => (
  <FlexBox
    justifyContent="space-between"
    position="absolute"
    bottom={0}
    width={1}
    zIndex={1}
  >
    <Box padding="0 1em">
    </Box>
    <Box padding="1em">

    </Box>
  </FlexBox>
);

const SubHeading = (props) => <Text color="secondary" textAlign="center" fontSize="h3" {...props} />

const TITLE = "Machine-Learned Prediction Equilibrium for Dynamic Traffic Assignment"
const PRESENTER = "Michael Markl"

const CustomSlide = ({ section, intro = false, children }) => {
  if (!intro) {
    return <Slide><FlexBox width="1" flexDirection="row" justifyContent="space-between" borderBottom="1px solid black">
      <Text fontFamily="head" fontSize="head" margin="0px" padding="0px" style={{ letterSpacing: "-.5px" }}>
        {TITLE}<span style={{ padding: "0 10px" }}> — </span>{section}
      </Text>
      <Text fontFamily="head" fontSize="head" margin="0px" padding="0px" style={{ letterSpacing: "-.5px" }}>{PRESENTER}</Text>
    </FlexBox>
      {children}
    </Slide>
  }

  return <Slide>
    <Stepper values={['hide']} alwaysVisible>
      {(value, step, isActive) => {
        const bigSectionSpringRef = useRef()
        const bigSectionSpring = useSpring({ ref: bigSectionSpringRef, to: isActive ? { opacity: 0 } : { opacity: 1 } })
        const slideSpringRef = useRef()
        const slideSpring = useSpring({ ref: slideSpringRef, to: isActive ? { opacity: 1 } : { opacity: 0 } })

        useChain(isActive ? [bigSectionSpringRef, slideSpringRef] : [slideSpringRef, bigSectionSpringRef])
        return <>
          <FlexBox width="1" flexDirection="row" justifyContent="space-between" borderBottom="1px solid black">
            <Text fontFamily="head" fontSize="head" margin="0px" padding="0px" style={{ letterSpacing: "-.5px" }}>
              {TITLE}
              <animated.span style={slideSpring}>
                <span style={{ padding: "0 10px" }}> — </span>
                {section}
              </animated.span>
            </Text>
            <Text fontFamily="head" fontSize="head" margin="0px" padding="0px" style={{ letterSpacing: "-.5px" }}>{PRESENTER}</Text>
          </FlexBox>

          <FlexBox width={1} bottom={0} top={0} justifyContent="center" alignItems="center" zIndex={-1} position="absolute">
            <Heading>
              <animated.div style={bigSectionSpring}>{section}</animated.div>
            </Heading>
          </FlexBox>
          <animated.div style={slideSpring}>
            {children}
          </animated.div>
        </>
      }}
    </Stepper>

  </Slide>
}

const Presentation = () => (
  <Deck theme={theme} template={template}>
    <Slide>
      <Heading>{TITLE}</Heading>
      <Text className="authors" textAlign="center" fontSize="h2">Lukas Graf<sup>1</sup>, Tobias Harks<sup>1</sup>, Kostas Kollias<sup>2</sup>, and Michael Markl<sup>1</sup>
        <div style={{ fontSize: "0.8em", margin: "2em 0", display: "flex", justifyContent: "center" }}><span style={{ width: "300px" }}><b>1</b>: University of Augsburg</span><span style={{ width: "300px" }}><b>2</b>: Google</span></div>
      </Text>
    </Slide>

    <CustomSlide intro section="I. The Flow Model">
      <SubHeading textAlign="left">The Physical Flow Model</SubHeading>
      <Box>
        <div>
          <Box style={{ float: "right" }}>
            <Example1Svg />
          </Box>
          <Text style={{ margin: "0 32px", padding: "0" }}>We are given</Text>
          <UnorderedList style={{ margin: "0 32px" }}>
            <ListItem>a finite, directed graph {Tex`G=(V,E)`},</ListItem>
            <ListItem>edge travel times {Tex`\tau_e > 0`} and edge capacities {Tex`\nu_e> 0`} for {Tex`e\in E`}, and</ListItem>
            <ListItem>commodities {Tex`i\in I`} each with source and sink nodes {Tex`s_i, t_i\in V`} and <br />a network inflow rate {Tex`u_i: \mathbb R_{\geq 0} \to \mathbb R_{\geq 0}`}.</ListItem>
          </UnorderedList>
          <Appear><Definition>
            A <i>dynamic flow</i> {Tex`f=(f^+, f^-)`} consists of
            <UnorderedList style={{ margin: "0" }}>
              <ListItem>edge inflow rates {Tex`f^+_{i,e}:\mathbb R_{\geq 0}\to \mathbb R_{\geq 0}`} for {Tex`i\in I, e\in E`} and</ListItem>
              <ListItem>edge outflow rates {Tex`f^-_{i,e}: \mathbb R_{\geq 0}\to \mathbb R_{\geq 0}`} for {Tex`i\in I, e\in E`}.</ListItem>
            </UnorderedList>
          </Definition></Appear>
          <Appear><Notation>
            {Tex`f_e^+ \coloneqq \sum_{i\in I} f_{i,e}^+,`}
            <Appear tagName='span'>{Tex`~~f_e^- \coloneqq \sum_{i\in I} f_{i,e}^-,`}</Appear>
            <Appear tagName='span'>{Tex`~~q_e(\theta) \coloneqq \int_0^\theta f^+_e(z) - f^-_e(z+\tau_e) \,\mathrm dz`}</Appear>
          </Notation></Appear>
          <Appear><Definition>
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
          </Definition></Appear>

        </div>
      </Box>
    </CustomSlide>

    <CustomSlide section="I. The Flow Model">
      <SubHeading textAlign="left">The Behavioral Model</SubHeading>
      <Box>
        <UnorderedList margin="0 32px">
          <Appear><ListItem>The <i>exit time</i> when entering edge {Tex`e`} at time {Tex`\theta`} is given by {Tex`T_e(\theta)\coloneqq \theta + \tau_e + \frac{q_e(\theta)}{\nu_e}`}</ListItem></Appear>
          <Appear><ListItem>Each commodity {Tex`i\in I`} is equipped with a set of <i>predictors</i> {BTex`
          \hat q_{i,e} : \mathbb R_{\geq0} \times \mathbb R_{\geq 0} \times C(\mathbb R_{\geq0}, \mathbb R_{\geq0})^{E} \to \mathbb R_{\geq 0},
          \quad
          (\theta, \bar\theta, q)\mapsto\hat q_{i,e}(\theta; \bar\theta; q),`}
            where {Tex`\hat q_{i,e}(\theta; \bar\theta; q)`} describes the <i>predicted queue length </i>
            of edge {Tex`e`} at time {Tex`\theta`} as predicted at time {Tex`\bar\theta`} using the historical queue functions {Tex`q`}.</ListItem></Appear>
          <Appear><ListItem>The <i>predicted exit time</i> when entering an edge {Tex`e`} at time {Tex`\theta`} is given by {Tex`\hat T_{i,e}(\theta; \bar\theta; q)\coloneqq \theta + \tau_e + \frac{\hat q_{i,e}(\theta; \bar\theta, q)}{\nu_e}`}.</ListItem></Appear>
          <Appear><ListItem>The <i>predicted exit time</i> when entering a path {Tex`P=(e_1, \dots, e_k)`} at time {Tex`\theta`} is given by
            {BTex`\hat T_{i,P}(\theta; \bar\theta; q)
            \coloneqq \left(\hat T_{e_k}(\,\boldsymbol{\cdot}\,;\bar\theta;q) \circ \cdots \circ \hat T_{e_1}(\,\boldsymbol{\cdot}\,;\bar\theta;q)\right)(\theta).
            `}
          </ListItem></Appear>
          <Appear><ListItem>
            The <i>predicted earliest arrival</i> at {Tex`t_i`} when starting at time {Tex`\theta`} at {Tex`v`} is given by
            {BTex`\hat l_{i,v}(\theta; \bar\theta; q)
            \coloneqq \min_{P\text{ simple } v\text{-}t_i\text{-path}} \hat T_{i,P}(\theta;\bar\theta;q).
            `}
          </ListItem></Appear>
        </UnorderedList>
        <Appear><Definition>
          A pair {Tex`(\hat q, f)`} of predictors {Tex`\hat q = (\hat q_{i,e})_{i\in I, e\in E}`} and
          a dynamic flow {Tex`f`} is a <i>dynamic prediction equilibrium (DPE)</i>, if for all edges {Tex`e=vw`} and all {Tex`\theta \geq 0`} it holds that
          {BTex`
              f^+_{i,e}(\theta) > 0 \implies \hat l_{i,v}(\theta;\theta; q) \leq \hat l_{i,w}(\hat T_{i,e}( \theta;\theta; q ); \theta; q).
          `}
        </Definition></Appear>
      </Box>
    </CustomSlide>

    <CustomSlide intro section="II. Existence of DPE">
      <SubHeading textAlign="left">Example for Nonexistence</SubHeading>
      <Example>For the network to the right, we define
        <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between' }}><div>
          <UnorderedList>
            <Appear><ListItem>{Tex`\tau_e=1`} for all {Tex`e\in E`},</ListItem></Appear>
            <Appear><ListItem>{Tex`\nu_{st} = 1`}, {Tex`\nu_{sv} = \nu_{vt} = 2,`}</ListItem></Appear>
            <Appear><ListItem>a single commodity with network inflow rate {Tex`u \equiv 2`},</ListItem></Appear>
            <Appear><ListItem>{Tex`
            \hat q_e(\theta;\bar\theta; q) \coloneqq \begin{cases}
                q_e(\bar\theta),& \text{if $q_e(\bar\theta) < 1$}, \\
                2,              & \text{otherwise.}
            \end{cases}
        `}</ListItem></Appear>
          </UnorderedList>
        </div>
          <div style={{ height: '200px' }}><Example2Svg /></div>
        </div>
        <Appear>Starting from time {Tex`\theta = 1`}, there is no possible equilibrium flow split.</Appear>
      </Example>
      <Appear><Question>When do dynamic prediction equilibria exist?</Question></Appear>
    </CustomSlide>

    <CustomSlide section="II. Existence of DPE">
      <SubHeading textAlign="left">Sufficient Conditions for the Existence of DPEs</SubHeading>
      <Appear><Definition>A predictor {Tex`\hat q_{i,e}`} is <i>continuous</i>, if {BTex`
      \hat q_{i,e} : \mathbb R_{\geq0} \times \mathbb R_{\geq 0} \times C(\mathbb R_{\geq0}, \mathbb R_{\geq0})^{E} \to \mathbb R_{\geq 0},
      `} is continuous from the product topology,
        where all {Tex` C(\mathbb R_{\geq0}, \mathbb R_{\geq0})`} are equipped with the topology induced by the uniform norm,
        to {Tex`\R_{\geq 0}`}.
      </Definition></Appear>
      <Appear><Definition>
        A predictor {Tex`\hat q_{i,e}`} is <i>oblivious</i>, if for all {Tex`\bar\theta \in\mathbb R_{\geq0}`} it holds {Tex`
        \quad\forall q,q'\colon\quad
    q_{\hspace{.07em}\vert\hspace{.07em}[0, \bar\theta]^E} = q'_{\hspace{.07em}\vert\hspace{.07em}[0, \bar\theta]^E}
    \implies
    \hat q_{i,e}(\,\boldsymbol{\cdot}\,;\bar\theta;q)=\hat q_{i,e}(\,\boldsymbol{\cdot}\,;\bar\theta;q').
        `}
      </Definition></Appear>

      <Appear><Definition>
        A predictor {Tex`\hat q_{i,e}`} <i>respects FIFO</i>, if {Tex`\hat T_{i,e}(\,\boldsymbol{\cdot}\,;\bar\theta, q)`} is non-decreasing
        for all {Tex`\bar\theta\in \R_{\geq0}`} and {Tex` q\in C(\mathbb R_{\geq0},\mathbb R_{\geq0})^{E}`}.
      </Definition></Appear>

      <Appear><Theorem>
        If all network inflow rates {Tex`u_i`} are bounded and all predictors {Tex`\hat q_{i, e}`} are
        continuous, oblivious, and respect FIFO, then
        there exists a dynamic prediction equilibrium {Tex`(\hat q, f)`}.
      </Theorem></Appear>

    </CustomSlide>

    <CustomSlide intro section="III. Applied Predictors">
      <SubHeading textAlign="left">Applied Predictors</SubHeading>
      <div style={{
        position: 'relative', top: '-50px',
        marginLeft: "1050px", width: "200px", textAlign: "center",
        fontFamily: "'Open Sans'", fontSize: theme.fontSizes.text
      }}>Compatible with Existence-Theorem</div>
      <UnorderedList style={{ position: 'relative', top: '-50px' }}>
        <PredictorListItem text={<>
          <i>The Zero-Predictor </i>{Tex`\hat q^{\text{Z}}_{i,e}(\theta;\bar\theta;q) \coloneqq 0`}.<br />
          <Appear><p>Predicted shortest paths always remain the same.</p></Appear>
        </>} figure={(minimize) => <ZeroPredictorSvg minimize={minimize} />} compatible />
        <PredictorListItem text={<>
          <i>The constant predictor </i>{Tex`\hat q^{\text{C}}_{i,e}(\theta;\bar\theta;q) \coloneqq q_e(\bar\theta)`}.<br />
          <Appear><p>Assumes the current conditions for the future.</p></Appear>
        </>} figure={minimize => <ConstantPredictorSvg minimize={minimize} />} compatible />
        <PredictorListItem text={<>
          <i>The linear predictor </i>{Tex`\hat q^{\text{L}}_{i,e}(\theta;\bar\theta;q) \coloneqq 
          \left( q_e(\bar \theta)+\partial_-q_e(\bar \theta)\cdot \min\{ \theta-\bar\theta, H \} \right)^+
          `}.
          <Appear><p>Not continuous in {Tex`\bar\theta`} whenever {Tex`\partial_-q_e`} jumps.</p></Appear>
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
    </CustomSlide>

    <CustomSlide intro section="IV. Computational Study">
      <SubHeading textAlign="left">Extension-based Simulation</SubHeading>
      <UnorderedList>
        <Appear><ListItem>Approximate a DPE by rerouting agents in discrete time intervals {Tex`\bar\theta_k = k\cdot \varepsilon`}.</ListItem></Appear>
        <Appear><ListItem>We assume that the network inflow rates are piecewise constant with finite jumps</ListItem></Appear>
        <Appear><ListItem>The extension procedure for one routing interval {Tex`(\bar\theta_k,\bar\theta_{k+1})`}:
          <div style={{ width: '1200px' }}>
            <ThemeProvider theme={{ size: { width: '1200px' } }}>
              <OrderedList style={{ backgroundColor: 'white', border: '1px solid lightgray', fontFamily: '' }}>
                <Appear><ListItem>Gather predictions for {Tex`\bar\theta_k`}</ListItem></Appear>
                <Appear><ListItem>Compute active edges at time {Tex`\bar\theta_k`}</ListItem></Appear>
                <Appear><ListItem><Code>while </Code>{Tex`H < \bar\theta_{k+1}`}<Code> do:</Code></ListItem></Appear>
                <Appear><ListItem><Code>    </Code>Compute maximal {Tex`H'\leq\bar\theta_{k+1}`} such that {Tex`b_{i,v}^-(\theta)\coloneqq \sum_{e\in\delta_{v}^-} f_{i,e}^-(\theta) + u_i(\theta)\cdot\mathbf{1}_{v=s_i}`} is constant on {Tex`(H, H')`} for all {Tex`v\in V, i\in I`}</ListItem></Appear>
                <Appear><ListItem><Code>    </Code>Equally distribute {Tex`b_{i,v}^-(\theta)`} to the active outgoing edges</ListItem></Appear>
                <Appear><ListItem><Code>    </Code>{Tex`H \leftarrow H'`}</ListItem></Appear>
              </OrderedList>
            </ThemeProvider>
          </div>
        </ListItem></Appear>
      </UnorderedList>
    </CustomSlide>

    <CustomSlide section="IV. Computational Study">
      <SubHeading textAlign="left">Comparing the Performance of Predictors</SubHeading>
      <UnorderedList>
        <Appear><ListItem>
          We monitor the average travel time of particles over multiple DPE simulations with varying inflow rates.
        </ListItem></Appear>
        <Appear><ListItem>
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
        </ListItem></Appear>

        <Appear>
          <ListItem>
            Simulations in real-world road traffic networks (centre of Tokyo, Sioux Falls) show
            <UnorderedList>
              <ListItem>the linear regression prediction is amongst the best predictors analyzed,</ListItem>
              <Appear><ListItem>the Zero-Predictor performs worst most of the time,</ListItem></Appear>
              <Appear><ListItem>the simulation is capable of computing DPEs in large-scale networks.</ListItem></Appear>
            </UnorderedList>
          </ListItem>
        </Appear>
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
              <Appear><ListItem>We formulated a mathematically concise model that generalizes existing unrealistic models.</ListItem></Appear>
              <Appear><ListItem>In this model, we proved the existence of equilibria under mild assumptions on the predictors.</ListItem></Appear>
              <Appear><ListItem>The framework allows the integration of arbitrary ML methods as predictors.</ListItem></Appear>
            </UnorderedList>
          </td>
          <td>
            <UnorderedList style={{ display: "inline-block" }}>
              <Appear><ListItem>Generalize the predictor's input to allow for other flow related data than past queues.</ListItem></Appear>
              <Appear><ListItem>Embed more advanced ML methods for traffic forecast into the simulation.</ListItem></Appear>
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
  return <Stepper values={[1, 2]} alwaysVisible>
    {(value, step, isActive) => {
      if (!value) {
        return Tex`
          \left(
            \bar\theta + j\delta,
            {\color{transparent} \left(
              \sum_{e' \in N(e)} 
                {\color{black} \sum_{i=0}^k a_{i,j}^{\color{transparent}e'}\cdot q_{e{\color{transparent}'}}(\bar\theta-i\delta) }
              \right)^+ } 
          \right)
      .`
      } else if (value == 1) {
        return Tex`
          \left(
            \bar\theta + j\delta,
            {\color{transparent} \left(
              {\color{black}
              \sum_{e' \in N(e)} 
                 \sum_{i=0}^k a_{i,j}^{e'}\cdot q_{e'}(\bar\theta-i\delta) }
              \right)^+ } 
          \right)
      .`
      } else {
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
    }}
  </Stepper>
}

const PredictorListItem = ({ text, figure, compatible }) => {
  return <Appear><ListItem>
    <div style={{ display: 'flex', flexDirection: 'row' }}>
      <div style={{ width: '700px', height: '100px' }}>{text}</div>
      <div style={{ height: '90px' }}><Minimizer>{figure}</Minimizer></div>
      <div style={{ height: '100px', display: 'flex', justifyContent: 'center', alignItems: 'center', marginLeft: '160px' }}>
        <Appear>{compatible ? '✔️' : '❌'}</Appear></div>
    </div>
  </ListItem></Appear>
}

const Minimizer = ({ children }) => {
  return <Stepper values={[true]} alwaysVisible>{
    (value, step, isActive) => {
      return children(value || false)
    }
  }
  </Stepper>
}

const ShowcaseFormula = ({ formula, text }) => {
  return <Stepper values={[true, false]}>
    {(value, step, isActive) => {
      return <ListItem style={{ display: value === false ? 'list-item' : 'block' }}>
        <div style={
          {
            display: 'flex', flexDirection: 'row', alignItems: 'center', height: '30px', transition: 'transform 0.2s',
            transform: value === false ? 'translateY(0px)' : 'translateY(20px)'
          }}>
          <div>{text}</div>
          <div style={{ paddingLeft: '15px', transition: 'transform 0.2s', transform: value === false ? 'scale(.5)' : 'scale(1)', transformOrigin: 'left' }}>{formula}</div>
        </div></ListItem>
    }}
  </Stepper>
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

ReactDOM.render(<Presentation />, document.getElementById('root'))
