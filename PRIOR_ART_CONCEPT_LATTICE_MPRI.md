# Prior-art note: Concept Lattices in Classification and Neural Learning

**Project:** Patent M/P/R/I classification pipeline\
**Prepared:** 26 November 2026\
**Purpose:** Descriptive prior-art note for inclusion in the
repository.\
**Scope:** Three selected references from an initial targeted search.
This is not an exhaustive search or legal opinion.

## 1. Executive summary

The selected references establish that the broad ideas of (1) using
Formal Concept Analysis (FCA) / concept lattices for classification and
(2) integrating lattice structure into neural learning have already been
explored.

The references do **not**, by themselves, establish that every possible
implementation of the proposed M/P/R/I patent-classification pipeline is
anticipated. That would require a claim-specific comparison and a
broader search.

## 2. Project context

The project aims to classify patent documents along four independent
functional dimensions:

-   **M --- Monitor / represent:** observe, represent, or update the
    state of a system.
-   **P --- Predict / simulate:** estimate future states, forecast, or
    evaluate scenarios.
-   **R --- Recommend / decide:** select or recommend actions, including
    decision support.
-   **I --- Intervene / control:** act on or control a physical or
    managed system.

The proposed input is patent title and abstract, with optional textual
descriptions of figures generated offline by a vision-language model.
The four labels are independent binary dimensions (Y/N/U), not
necessarily one exclusive four-way class. The proposed approach is to
extract concepts and relations from the patent corpus, derive a concept
lattice from training data, and integrate that structure into a
trainable classifier. Company identity is intended for downstream
analysis, not as a classifier feature.

## 3. Selected references

### 3.1 Shao et al. (2023) --- Graph neural networks induced by concept lattices for classification

**Bibliographic information**

Mingwen Shao, Zhiyong Hu, Weizhi Wu, and Huan Liu, "Graph neural
networks induced by concept lattices for classification," *International
Journal of Approximate Reasoning*, Volume 154, March 2023, pp. 262--276.

-   DOI: https://doi.org/10.1016/j.ijar.2023.01.001
-   Publisher page:
    https://www.sciencedirect.com/science/article/pii/S0888613X23000026

**Technical summary**

The paper proposes integrating concept-lattice theory into
graph-neural-network message passing. The lattice represents
generalization/specialization relationships among formal concepts and is
used as structural information for classification.

**Relevance**

This is a direct precedent for using a concept lattice within the
computation of a neural classifier, rather than treating the lattice
solely as a separate post-processing classifier. It weighs against
treating "combining a neural network with a concept lattice" as a novel
idea by itself.

**Distinction**

The work concerns its own GNN formulation and evaluation tasks. The
source reviewed does not establish the specific patent-document input,
four independent M/P/R/I outputs, VLM-generated patent-figure
descriptions, or downstream organizational inference proposed here.

### 3.2 Vemuri et al. (2026) --- Formal Concept Lattices are Good Semantic Scaffolds for Concept-Based Learning

**Bibliographic information**

Deepika Vemuri, Sayanta Adhikari, Ankit Saha, Krishn Vishwas Kher, and
Vineeth N. Balasubramanian, "Formal Concept Lattices are Good Semantic
Scaffolds for Concept-Based Learning," *Proceedings of the 43rd
International Conference on Machine Learning*, Proceedings of Machine
Learning Research, Volume 306, 2026, pp. 123653--123687.

-   Proceedings page: https://proceedings.mlr.press/v306/vemuri26a.html
-   arXiv preprint: https://arxiv.org/abs/2606.05471
-   PDF:
    https://raw.githubusercontent.com/mlresearch/v306/main/assets/vemuri26a/vemuri26a.pdf

**Technical summary**

The paper uses formal concept lattices as semantic scaffolds to guide
where concepts at different levels of generality should be learned
within a neural network. It aims to produce more interpretable,
hierarchically structured concept representations.

**Relevance**

This is directly relevant to using lattice structure to guide the
learning process itself, rather than only applying a lattice after
neural inference.

**Distinction**

The reviewed publication record does not establish the full M/P/R/I
patent-classification pipeline, the specific patent-corpus
concept-extraction method, or the optional VLM-based figure-description
path.

### 3.3 US10360506B2 --- General formal concept analysis (FCA) framework for classification

**Bibliographic information**

-   Title: "General formal concept analysis (FCA) framework for
    classification"
-   Patent: US10360506B2
-   Publication: US20170316318A1
-   Priority date shown by Google Patents: 23 July 2014
-   Grant publication date: 23 July 2019
-   Google Patents: https://patents.google.com/patent/US10360506B2/en

**Technical summary**

The patent describes building a context table from training examples and
attributes, appending class columns, and constructing an FCA
classification lattice. For a new input, the system forms an
attribute/context vector, selects voting nodes in the lattice, and
combines their votes to produce a classification. The disclosure
discusses weighting votes by shared attributes and by properties of the
nodes.

**Relevance**

This is a patent-document precedent for deriving an FCA classification
lattice from labeled training data and using the lattice to classify
unseen inputs.

**Distinction**

The disclosed approach is an FCA-based classification and voting
framework. It should not automatically be equated with a differentiable
neural network whose training objective or intermediate representations
are explicitly conditioned on a lattice.

**Legal-status caution**

The legal-status information displayed by Google Patents is not a legal
conclusion. This note does not independently verify current
enforceability, family status, prosecution history, or claim scope in
any jurisdiction.

## 4. Comparison with the proposed project

  --------------------------------------------------------------------------------------------
  Feature                     Shao et        Vemuri et      US10360506B2      Project concept
                              al. (2023)     al. (2026)                       
  --------------------------- -------------- -------------- ----------------- ----------------
  FCA / concept lattice       Yes            Yes            Yes               Yes, proposed

  Lattice used for            Yes            Supports       Yes               Intended
  classification                             concept                          
                                             learning                         

  Lattice integrated into     Integrated     Guides         Not established   Intended;
  neural computation/learning into GNN       hierarchical   as a              requires
                              message        neural concept differentiable    implementation
                              passing        learning       neural-learning   
                                                            mechanism in the  
                                                            reviewed          
                                                            disclosure        

  Patent title and abstract   Not            Not            Not specific to   Yes
  as input                    established    established    patent documents  

  Optional VLM descriptions   Not            Not            Not established   Yes
  of patent figures           established    established                      

  Four independent M/P/R/I    Not            Not            Not established   Yes
  outputs                     established    established                      

  Corpus-derived lattice for  Not            Not            Training context  Proposed
  this patent application     established    established    and class         
                                                            attributes build  
                                                            an FCA lattice    

  Downstream inference of     Not            Not            Not established   Separate
  company                     established    established                      objective
  capabilities/organization                                                   
  --------------------------------------------------------------------------------------------

"Not established" means the feature is not established by the sources
reviewed for this note; it does not mean no other prior art discloses
it.

## 5. Implications

The selected references indicate that these broad ideas are already
known:

-   FCA and concept lattices can be used for classification.
-   Lattices encode generalization/specialization relations among
    concepts.
-   Lattice structure can be integrated into graph-neural-network
    message passing.
-   Formal concept lattices can guide hierarchical semantic concept
    learning in neural networks.
-   FCA classification lattices can be constructed from training
    examples and class attributes and used to classify new inputs.

The project should therefore not rely on the general statement "we
combine a neural network with a Concept Lattice" as its distinguishing
contribution.

Potentially differentiating details to investigate---without assuming
they are novel or patentable---include the exact patent-corpus concept
construction, representation of functional roles and evidence spans,
integration mechanism for four independent M/P/R/I outputs, handling of
missing figure descriptions, and downstream capability inference.

## 6. Experimental status

The reported single-patent proof of concept used existing
TF-IDF/logistic-regression checkpoints and an experimental keyword
baseline. It did **not** demonstrate a newly trained neural network with
an integrated Concept Lattice, nor establish convergence of that
proposed architecture.

The POC reported that only one of three patent figures received a
syntactically validated VLM description. The single-patent comparison is
an integration smoke test, not evidence of classification accuracy or
generalization.

A suitable next experiment would compare on leakage-controlled
train/validation/test splits:

1.  A reproducible text-only baseline.
2.  A newly trained model that integrates the proposed lattice
    mechanism.
3.  If representative figure descriptions are available, a controlled
    text-only versus text-plus-description comparison.

The four labels should be evaluated independently. Report training and
validation loss curves, per-label precision/recall/F1 and
class-imbalance-aware metrics, plus held-out patent results. Ten
held-out patents can support an initial qualitative inspection, but not
a statistically strong generalization claim.

## 7. Search scope and limitations

This note records three selected references directly relevant to the
central lattice/classification idea. It is not exhaustive and does not
include a systematic search of all scientific databases, patent offices,
non-English publications, patent families, citations, prosecution files,
or all potentially relevant work.

No legal opinion is provided on novelty, inventive step, eligibility,
infringement, ownership, or freedom to operate. Before filing or making
a legal decision, a patent professional should compare the exact
proposed claims against the full text and legal status of relevant
references in the relevant jurisdictions.

## 8. Source links

1.  Shao et al. (2023), *Graph neural networks induced by concept
    lattices for classification*:
    https://doi.org/10.1016/j.ijar.2023.01.001
2.  Vemuri et al. (2026), *Formal Concept Lattices are Good Semantic
    Scaffolds for Concept-Based Learning*:
    https://proceedings.mlr.press/v306/vemuri26a.html
3.  US10360506B2, *General formal concept analysis (FCA) framework for
    classification*: https://patents.google.com/patent/US10360506B2/en
