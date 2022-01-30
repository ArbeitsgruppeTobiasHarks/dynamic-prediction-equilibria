/*! For license information please see deck.js.LICENSE.txt */
              \begin{cases}
              = u_i(\theta), & \text{if $v = s_i$}, \\
              = 0, & \text{if $v \notin \{s_i, t_i \}$}, \\
              \leq 0, & \text{if $v = t_i$}.
              \end{cases}`}),t.createElement(Lm,{text:"Queues operate at capacity:",formula:am`f_e^-(\theta) = \begin{cases}
            \nu_e,&\text{if $q_e(\theta - \tau_e) > 0$,} \\
            \min\{ f_e^+(\theta- \tau_e), \nu_e \}, &\text{otherwise.}
          \end{cases}`}),t.createElement(Lm,{text:"Capacity is split fairly:",formula:am`
                f_{i,e}^-(\theta) = f_e^-(\theta) \cdot \frac{f_{i,e}^+(\xi)}{f_e^+(\xi)}
                \quad\text{for $\xi\coloneqq \min\{\xi\leq\theta \mid \xi + \tau_e + \frac{q_e(\xi)}{\nu_e} = \theta \}$ with $f_e^+(\xi) > 0$}`}))))))),t.createElement(xm,{section:"I. The Flow Model"},t.createElement(wm,{textAlign:"left"},"The Behavioral Model"),t.createElement(Ba,null,t.createElement(Ns,{margin:"0 32px"},t.createElement(Sm,null,"The ",t.createElement("i",null,"exit time")," when entering edge ",rm`e`," at time ",rm`\theta`," is given by ",rm`T_e(\theta)\coloneqq \theta + \tau_e + \frac{q_e(\theta)}{\nu_e}`),t.createElement(Sm,null,"Each commodity ",rm`i\in I`," is equipped with a set of ",t.createElement("i",null,"predictors")," ",am`
          \hat q_{i,e} : \mathbb R_{\geq0} \times \mathbb R_{\geq 0} \times C(\mathbb R_{\geq0}, \mathbb R_{\geq0})^{E} \to \mathbb R_{\geq 0},
          \quad
          (\theta, \bar\theta, q)\mapsto\hat q_{i,e}(\theta; \bar\theta; q),`,"where ",rm`\hat q_{i,e}(\theta; \bar\theta; q)`," describes the ",t.createElement("i",null,"predicted queue length "),"of edge ",rm`e`," at time ",rm`\theta`," as predicted at time ",rm`\bar\theta`," using the historical queue functions ",rm`q`,"."),t.createElement(Sm,null,"The ",t.createElement("i",null,"predicted exit time")," when entering an edge ",rm`e`," at time ",rm`\theta`," is given by ",rm`\hat T_{i,e}(\theta; \bar\theta; q)\coloneqq \theta + \tau_e + \frac{\hat q_{i,e}(\theta; \bar\theta, q)}{\nu_e}`,"."),t.createElement(Sm,null,"The ",t.createElement("i",null,"predicted exit time")," when entering a path ",rm`P=(e_1, \dots, e_k)`," at time ",rm`\theta`," is given by",am`\hat T_{i,P}(\theta; \bar\theta; q)
            \coloneqq \left(\hat T_{e_k}(\,\boldsymbol{\cdot}\,;\bar\theta;q) \circ \cdots \circ \hat T_{e_1}(\,\boldsymbol{\cdot}\,;\bar\theta;q)\right)(\theta).
            `),t.createElement(Sm,null,"The ",t.createElement("i",null,"predicted earliest arrival")," at ",rm`t_i`," when starting at time ",rm`\theta`," at ",rm`v`," is given by",am`\hat l_{i,v}(\theta; \bar\theta; q)
            \coloneqq \min_{P\text{ simple } v\text{-}t_i\text{-path}} \hat T_{i,P}(\theta;\bar\theta;q).
            `)),t.createElement(Dm,null,"A pair ",rm`(\hat q, f)`," of predictors ",rm`\hat q = (\hat q_{i,e})_{i\in I, e\in E}`," and a dynamic flow ",rm`f`," is a ",t.createElement("i",null,"dynamic prediction equilibrium (DPE)"),", if for all edges ",rm`e=vw`," and all ",rm`\theta \geq 0`," it holds that",am`
              f^+_{i,e}(\theta) > 0 \implies \hat l_{i,v}(\theta;\theta; q) \leq \hat l_{i,w}(\hat T_{i,e}( \theta;\theta; q ); \theta; q).
          `))),t.createElement(xm,{intro:!0,section:"II. Existence of DPE"},t.createElement(wm,{textAlign:"left"},"Example for Nonexistence"),t.createElement(Fm,null,t.createElement("div",{style:{float:"right"}},t.createElement(tm,null)),"For the network to the right, we define",t.createElement(Ns,null,t.createElement(Sm,null,rm`\tau_e=1`," for all ",rm`e\in E`,","),t.createElement(Sm,null,rm`\nu_{st} = 1`,", ",rm`\nu_{sv} = \nu_{vt} = 2,`),t.createElement(Sm,null,"a single commodity with network inflow rate ",rm`u \equiv 2`,","),t.createElement(Sm,null,rm`
            \hat q_e(\theta;\bar\theta; q) \coloneqq \begin{cases}
                q_e(\bar\theta),& \text{if $q_e(\bar\theta) < 1$}, \\
                2,              & \text{otherwise.}
            \end{cases}
        `)),"Starting from time ",rm`\theta = 1`,", there is no possible equilibrium flow split."),t.createElement(Pm,null,"When do dynamic prediction equilibria exist?")),t.createElement(xm,{section:"II. Existence of DPE"},t.createElement(wm,{textAlign:"left"},"Sufficient Conditions for the Existence of DPEs"),t.createElement(Dm,null,"A predictor ",rm`\hat q_{i,e}`," is ",t.createElement("i",null,"continuous"),", if ",am`
      \hat q_{i,e} : \mathbb R_{\geq0} \times \mathbb R_{\geq 0} \times C(\mathbb R_{\geq0}, \mathbb R_{\geq0})^{E} \to \mathbb R_{\geq 0},
      `," is continuous from the product topology, where all ",rm` C(\mathbb R_{\geq0}, \mathbb R_{\geq0})`," are equipped with the topology induced by the uniform norm, to ",rm`\R_{\geq 0}`,"."),t.createElement(Dm,null,"A predictor ",rm`\hat q_{i,e}`," is ",t.createElement("i",null,"oblivious"),", if for all ",rm`\bar\theta \in\mathbb R_{\geq0}`," it holds ",rm`
        \quad\forall q,q'\colon\quad
    q_{\hspace{.07em}\vert\hspace{.07em}[0, \bar\theta]^E} = q'_{\hspace{.07em}\vert\hspace{.07em}[0, \bar\theta]^E}
    \implies
    \hat q_{i,e}(\,\boldsymbol{\cdot}\,;\bar\theta;q)=\hat q_{i,e}(\,\boldsymbol{\cdot}\,;\bar\theta;q').
        `),t.createElement(Dm,null,"A predictor ",rm`\hat q_{i,e}`," ",t.createElement("i",null,"respects FIFO"),", if ",rm`\hat T_{i,e}(\,\boldsymbol{\cdot}\,;\bar\theta, q)`," is non-decreasing for all ",rm`\bar\theta\in \R_{\geq0}`," and ",rm` q\in C(\mathbb R_{\geq0},\mathbb R_{\geq0})^{E}`,"."),t.createElement(Bm,null,"If all network inflow rates ",rm`u_i`," are bounded and all predictors ",rm`\hat q_{i, e}`," are continuous, oblivious, and respect FIFO, then there exists a dynamic prediction equilibrium ",rm`(\hat q, f)`,".")),t.createElement(xm,{intro:!0,section:"III. Applied Predictors"},t.createElement(wm,{textAlign:"left"},"Applied Predictors"),t.createElement("div",{style:{position:"relative",top:"-50px",marginLeft:"1050px",width:"200px",textAlign:"center",fontFamily:"'Open Sans'",fontSize:_m.fontSizes.text}},"Compatible with Existence-Theorem"),t.createElement(Ns,{style:{position:"relative",top:"-50px"}},t.createElement(Im,{text:t.createElement(t.Fragment,null,t.createElement("i",null,"The Zero-Predictor "),rm`\hat q^{\text{Z}}_{i,e}(\theta;\bar\theta;q) \coloneqq 0`,".",t.createElement("br",null),t.createElement($o,null,t.createElement("p",null,"Predicted shortest paths always remain the same."))),figure:e=>t.createElement(mm,{minimize:e}),compatible:!0}),t.createElement(Im,{text:t.createElement(t.Fragment,null,t.createElement("i",null,"The constant predictor "),rm`\hat q^{\text{C}}_{i,e}(\theta;\bar\theta;q) \coloneqq q_e(\bar\theta)`,".",t.createElement("br",null),t.createElement($o,null,t.createElement("p",null,"Assumes the current conditions for the future."))),figure:e=>t.createElement(pm,{minimize:e}),compatible:!0}),t.createElement(Im,{text:t.createElement(t.Fragment,null,t.createElement("i",null,"The linear predictor "),rm`\hat q^{\text{L}}_{i,e}(\theta;\bar\theta;q) \coloneqq 
          \left( q_e(\bar \theta)+\partial_-q_e(\bar \theta)\cdot \min\{ \theta-\bar\theta, H \} \right)^+
          `,".",t.createElement($o,null,t.createElement("p",null,"Not continuous in ",rm`\bar\theta`," whenever ",rm`\partial_-q_e`," jumps."))),figure:e=>t.createElement(gm,{minimize:e}),compatible:!1}),t.createElement(Im,{text:t.createElement(t.Fragment,null,t.createElement("i",null,"The regularized linear predictor "),t.createElement("br",null),t.createElement("div",{style:{textAlign:"center"}},rm`\hat q_{i,e}^{\text{RL}}(\theta;\bar\theta; q) \coloneqq
\Big( q_e(\bar\theta) + \frac{q_e(\bar\theta) - q_e(\bar\theta - \delta)}{\delta} \cdot \min\{ \theta - \bar\theta, H \} \Big)^+
      .`)),figure:e=>t.createElement(bm,{minimize:e}),compatible:!0}),t.createElement(Im,{text:t.createElement(t.Fragment,null,t.createElement("i",null,"The linear regression predictor ")," ",rm`\hat q_{i,e}^{\text{ML}}`," linearly interpolates the points ",t.createElement("br",null),t.createElement("div",{style:{textAlign:"center"}},t.createElement(Om,null))),figure:e=>t.createElement(Em,{minimize:e}),compatible:!0})))),Om=()=>t.createElement(Wo,{values:[1,2],alwaysVisible:!0},((e,t,n)=>e?1==e?rm`
          \left(
            \bar\theta + j\delta,
            {\color{transparent} \left(
              {\color{black}
              \sum_{e' \in N(e)} 
                 \sum_{i=0}^k a_{i,j}^{e'}\cdot q_{e'}(\bar\theta-i\delta) }
              \right)^+ } 
          \right)
      .`:rm`
          \left(
            \bar\theta + j\delta,
             \left(  
              \sum_{e' \in N(e)} 
                 \sum_{i=0}^k a_{i,j}^{e'}\cdot q_{e'}(\bar\theta-i\delta)
              \right)^+
          \right)
      .`:rm`
          \left(
            \bar\theta + j\delta,
            {\color{transparent} \left(
              \sum_{e' \in N(e)} 
                {\color{black} \sum_{i=0}^k a_{i,j}^{\color{transparent}e'}\cdot q_{e{\color{transparent}'}}(\bar\theta-i\delta) }
              \right)^+ } 
          \right)
      .`)),Im=({text:e,figure:n,compatible:r})=>t.createElement($o,null,t.createElement(Sm,null,t.createElement("div",{style:{display:"flex",flexDirection:"row"}},t.createElement("div",{style:{width:"700px",height:"100px"}},e),t.createElement("div",{style:{height:"90px"}},t.createElement(Rm,null,n)),t.createElement("div",{style:{height:"100px",display:"flex",justifyContent:"center",alignItems:"center",marginLeft:"160px"}},t.createElement($o,null,r?"✔️":"❌"))))),Rm=({children:e})=>t.createElement(Wo,{values:[!0],alwaysVisible:!0},((t,n,r)=>e(t||!1))),Lm=({formula:e,text:n})=>t.createElement(Wo,{values:[!0,!1]},((r,a,i)=>t.createElement(Sm,{style:{display:!1===r?"list-item":"block"}},t.createElement("div",{style:{display:"flex",flexDirection:"row",alignItems:"center",height:"30px",transition:"transform 0.2s",transform:!1===r?"translateY(0px)":"translateY(20px)"}},t.createElement("div",null,n),t.createElement("div",{style:{paddingLeft:"15px",transition:"transform 0.2s",transform:!1===r?"scale(.5)":"scale(1)",transformOrigin:"left"}},e))))),Mm=({children:e})=>t.createElement(Ba,{margin:"32px",style:{fontSize:_m.fontSizes.text,fontFamily:"Open Sans"}},t.createElement("span",null,t.createElement("i",null,"Notation. ")),e),Pm=({children:e})=>t.createElement(Ba,{margin:"32px",style:{fontSize:_m.fontSizes.text,fontFamily:"Open Sans"}},t.createElement("span",{style:{color:_m.colors.secondary}},t.createElement("b",null,"Question. ")),e),Dm=({children:e})=>t.createElement(Ba,{margin:"32px",style:{fontSize:_m.fontSizes.text,fontFamily:"Open Sans"}},t.createElement("span",{style:{color:_m.colors.secondary}},t.createElement("b",null,"Definition. ")),e),Fm=({children:e})=>t.createElement(Ba,{margin:"32px",style:{fontSize:_m.fontSizes.text,fontFamily:"Open Sans"}},t.createElement("span",{style:{color:_m.colors.secondary}},t.createElement("b",null,"Example. ")),e),Bm=({children:e})=>t.createElement(Ba,{margin:"32px",style:{fontSize:_m.fontSizes.text,fontFamily:"Open Sans"}},t.createElement("span",{style:{color:_m.colors.secondary}},t.createElement("b",null,"Theorem. ")),t.createElement("i",null,e));r.render(t.createElement(Cm,null),document.getElementById("root"))})()})();
//# sourceMappingURL=deck.js.map