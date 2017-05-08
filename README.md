# Background

Traditional models of economics, psychology, and other sciences assume rational models of agents. Individu- als in the real world, however, rely on heuristics with biases which appear to result in temporally inconsistent choices, false beliefs, and seemingly suboptimal planning (Kahneman and Tversky, 1979). Often, humans make choices that change over time, do not serve their self-interest, and conflict with their stated preferences.

An important application of machine learning and probabilistic programming is to essentially understand the underlying cognitive processes that characterize and explain these seemingly ‘sub-optimal’ choices (Evans et al., 2015). The need to understand the latent factors influencing these choices has spurred research in economics, psychology, and marketing, with substantial work done on modeling and predicting future human behavior based on past behavior and certain latent factors. As machine learning and AI continue to find applicability in automating personal tasks and in interacting with people, it becomes important that these models are able to understand the nuances in human choices to better predict future behavior. To this end, developing probabilistic models of future choices based on past behavior has become a topical and important problem.

Research in economics and marketing has led to several computational models that attempt to model human behavior from their exhibited choices (Szenberg et al., 2009), however, these models haven’t stemmed from psychological models of human choice. Concurrently, several psychological models of human choice exist, however, these haven’t been applied to the problem of behavioral model inference from observed choices (Busemeyer and Johnson, 2005; Train, 2003; Shenoy and Angela, 2013). Alternatively, computational mod- els in Inverse Reinforcement Learning (Ng et al., 2000) and Bayesian Inverse Planning (Baker et al., 2009) attempt to infer preferences by inverting a rational decision-making model based on a utility function (Rus- sell et al., 1995). These models, however, haven’t been able to capture the full spectrum of the seeming irrationalities underlying human decision models. Their shortcomings, traditionally, have stemmed from ignoring the latent factors spurring the bounded rationality, usually by modeling these inconsistencies as noise (Kim et al., 2014; Zheng et al., 2014).

Recent developments (Evans et al., 2015; Jern et al., 2017) have attempted to correct these simplifications by explicitly accounting for the biases and heuristics developed in psychological models of human choice (Kahneman and Tversky, 1979). Progress on this frontier has included structural modifications to the utility functions that incorporate the biases and heuristics. For example, hyperbolic discounting (Ainslie, 2001) is used to capture the tendency for temporally inconsistent choices while uncertainty and false beliefs affecting choices are modeled through probability distributions over knowledge of the state of the world, instead of assuming agents fully know the current state of the world (Baker and Tenenbaum, 2014).

Despite the success in explaining inconsistencies in choices as compared to human explanations, there remains substantial work to be done on defining expressive and representative computational frameworks. These frameworks need to be expressive and reliable enough to help AI, and human researchers, better understand human behavior.

# Attention Modeling

A particularly interesting aspect of behavior subject to decision and choice theory is human attention. The ‘choice’ of whether to continue focusing on the current task at hand or to switch to another task is sub- ject to several latent factors. For example, switching from the current task of work to leisure (e.g. social media) could mean the agent is done with their work, but could alternatively be due to distractions or fatigue.

Developing probabilistic models that are able to infer what underlying cognitive state an agent is in based on the observed attention patterns an agent is displaying can prove to be beneficial in predicting whether the agent will continue working on the task at hand or switch tasks. Due to the myriad of layered and latent factors influencing attention models, exact and accurate models are hard to develop, however approximate models that address several important latent variables are still very beneficial.

The potential implications of attention-aware computational models are far-reaching. For example, models capable of predicting whether an agent will switch from the required task at hand could find usage in pre- dicting when drivers will become distracted and switch tasks from driving to using their phones. Models with such capabilities can then alert the driver or take action, potentially avoiding life-threatening accidents. Moreover, in an attention economy (Davenport and Beck, 2013) where many users report reduced produc- tivity due to constant distractions, such models could be used in anti-distraction applications that aim to improve productivity by ensuring people remained focused on their primary tasks. Finally, by analyzing the predictions of these computational models and their accuracy, we can further learn more about the charac- teristics and properties of the latent factors affecting task-switching behavior, allowing us to improve our existing knowledge on human attention. These high-impact ramifications serve to motivate the design and analysis of such predictive models.

To this end, this thesis project will attempt to develop a generative framework that models some latent factors that affect attention and result in task-switching. Inference over these models will be performed to recover/infer the underlying cognitive states an agent is in with the results of the inference used to determine the probability of an agent switching tasks.

# Proposed Approach

Human attention involves choices on which current task at hand to focus on. Distractions, temptations and temporal inconsistencies influence agents when choosing which tasks to focus on. For my project, I intend to focus particularly on probabilistically modeling human attention and subsequently inferring agents’ cognitive states and how predictive these states are of future task-switching. To address the aforementioned problem, this thesis will build on the approaches of agent-based models (Ng et al., 2000; Baker and Tenenbaum, 2014; Baker et al., 2009; Evans et al., 2015) by applying them to the task of predicting task switching. The generative models will make these predictions by first inferring the latent cognitive state of an agent given their previous attention patterns and choices.

To approximately infer an agent’s cognitive states from their observed choices, several latent factors in hu- man decision frameworks need to be modeled. Consequently, a substantial portion of this project will be dedicated to identifying the salient features affecting human attention models and designing models that account for them. To do this, this project will probe psychological models of attention, distraction, and procrastination e.g. (Ariely and Wertenbroch, 2002; Duckworth et al., 2016) along with behavioral models of attention in seeking information e.g. (Kurzban et al., 2013; Pirolli and Card, 1999). From these models, a select set of underlying features will be modeled in the the probabilistic models to allow for more accurate inferences to be performed. However, due to the complex nature of Bayesian inferences, the scope of this thesis will be narrowed to account for only a few of the most predictive factors of task-switching.

To test the effectiveness of this approach, a narrow class of generative models will be analysed and compared against a naive baseline, for example, constantly predicting that an agent will continue to focus on the task they are currently focusing on. This comparison will be done on a real-life dataset of tasks attended to by people obtained from RescueTime, an application that tracks a users behaviour on their computer across different applications. RescueTime tracks attention across a set of ‘productive’ and ‘distracting’ actions. Ac- tivities such as software development, research and document typing are labeled as productive actions while social media, instant messaging and watching movies are examples of distracting actions. The logged data by RescueTime for a set of participants, including myself, will be collected and processed into time-series data depicting the tasks the participant attended to. This input will then be fed into the developed generative models. The effectiveness of each model will be measured via the model’s ability to correctly predict when an agent is about to switch tasks based on the inferred cognitive state of the agent, for example, whether the agent is fatigued or not.

The input to the models will be only the labeled time-series data of tasks the agent is focusing on. As such, low-level factors such as lighting, location, mental/physical well-being, etc of the agents will be ignored. The task will strictly be to predict task-switching based on information logged by RescueTime (e.g. time of day, time elapsed since activity started, hours spent working during that day, etc).

In essence, this project will attempt to explore three questions:

1) Can human attention be modeled as a sequential decision making problem and tackled by agent-based modelling?
2) Does explicitly modelling for theoretical factors affecting attention increase the predictive power of these models, and which factors possess more predictive power than others?
3) Finally, are these models able to make plausible inferences over the latent cognitive state of the agent?


#References
George Ainslie. Breakdown of will. Cambridge University Press, 2001.

Dan Ariely and Klaus Wertenbroch. Procrastination, deadlines, and performance: Self-control by precommitment. Psychological science, 13(3):219–224, 2002.

Chris L Baker and Joshua B Tenenbaum. Modeling human plan recognition using bayesian theory of mind. Plan, activity, and intent recognition: Theory and practice, pages 177–204, 2014.

Chris L Baker, Rebecca Saxe, and Joshua B Tenenbaum. Action understanding as inverse planning. Cognition, 113(3):329–349, 2009.

Jerome R Busemeyer and Joseph G Johnson. Micro-process Models of Decision Making. Brand, 2:1–39, 2005. ISSN 01635182. URL http://neuroeconomics-summerschool.stanford.edu/pdf/BUSEMEYER2.pdf. 

Thomas H Davenport and John C Beck. The attention economy: Understanding the new currency of business.
Harvard Business Press, 2013.

Angela L Duckworth, Tamar Szab ́o Gendler, and James J Gross. Situational strategies for self-control.
Perspectives on Psychological Science, 11(1):35–55, 2016.

Owain Evans, Andreas Stuhlmu ̈ller, and Noah D Goodman. Learning the preferences of bounded agents. NIPS Workshop on Bounded Optimality, pages 16–22, 2015. URL http://web.mit.edu/owain/www/nips-workshop-2015-website.pdf.

Alan Jern, Christopher G Lucas, and Charles Kemp. People learn other people’s preferences through inverse decision-making. PhD thesis, Rose-Hulman Institute of Technology, 2017.

D Kahneman and A Tversky. Prospect theory - Analysis of decision under risk, 1979. ISSN 0012-9682.

Dongho Kim, Catherine Breslin, Pirros Tsiakoulis, Milica Gaˇsi ́c, Matthew Henderson, and Steve Young. Inverse reinforcement learning for micro-turn management. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, (September):328–332, 2014. ISSN 19909772.

Robert Kurzban, Angela Duckworth, Joseph W Kable, and Justus Myers. An opportunity cost model of subjective effort and task performance. Behavioral and Brain Sciences, 36(06):661–679, 2013.

Andrew Y Ng, Stuart J Russell, et al. Algorithms for inverse reinforcement learning. In Icml, pages 663–670, 2000.

Peter Pirolli and Stuart Card. Information foraging. Psychological review, 106(4):643, 1999.

Stuart Russell, Peter Norvig, and Artificial Intelligence. A modern approach. Artificial Intelligence. Prentice- Hall, Egnlewood Cliffs, 25:27, 1995.

Pradeep Shenoy and J Yu Angela. Rational preference shifts in multi-attribute choice: What is fair? In CogSci, 2013.

Michael Szenberg, Lall Ramrattan, and Aron A. Gottesman. Samuelsonian Economics and the Twenty-First Century. 2009. ISBN 9780191711480. doi: 10.1093/acprof:oso/9780199298839.001.0001.

Kenneth E. Train. Discrete Choice Methods with Simulation. Cambridge University Press, pages 1–388, 2003. ISSN 08981221. doi: 10.1017/CBO9780511753930. URL http://ebooks.cambridge.org/ref/id/CBO9780511753930.

Jiangchuan Zheng, Siyuan Liu, and Lionel M Ni. Robust Bayesian Inverse Reinforcement Learning with Sparse Behavior Noise. AAAI, pages 2198–2205, 2014.
