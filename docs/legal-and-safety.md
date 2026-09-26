# Scope, safety, and legal notes

This page explains the project's purpose and intended boundary. It is not legal
advice. Laws and platform rules change, so check the current rules that apply to
your location and use case.

## Purpose

This is a research project. It exists to study how AI provenance signals behave:
how robust visible labels, invisible watermarks, and C2PA or metadata
disclosures are, how they fail, and how reliably they can be identified. The
same code gives people control over provenance marks on content they generated
or edited themselves, and helps correct false AI labels.

The project covers:

- visible AI generation labels;
- invisible provenance watermarks;
- C2PA and metadata based AI disclosures.

## No warranty and no liability

The software is provided under the Apache License 2.0, "AS IS", without
warranties or conditions of any kind (section 7). The authors and contributors
are not liable for any damages arising from its use (section 8). They do not
monitor, control, or endorse what anyone does with the software or with its
outputs. Each user is solely responsible for determining whether a given use is
lawful where they are and where the content is published.

A license disclaimer governs the relationship between the authors and users.
It does not override public law. Where a jurisdiction restricts removing AI
labels, or supplying tools that do so, that restriction applies regardless of
this notice; see the table below.

## Out of scope

The project does not provide automatic removal for marks that protect a third
party's paid or copyrighted asset, including:

- stock agency previews;
- marketplace and classifieds marks;
- tiled overlays used to gate a purchase;
- artist protection systems such as Nightshade or Glaze.

It also does not add templates for regulator-defined disclosure icons, such as
the EU icons in Annex 1 of the Code of Practice on marking and labelling of
AI-generated content, which deployers use to disclose deepfakes.

The `erase` command is a generic region tool. Users are responsible for having
the right to edit the selected content.

## What removal does not prove

Removing a local signal does not:

- prove that an image is human made;
- remove server side generation history;
- anonymize the generating account;
- defeat every statistical AI detector;
- guarantee that a provider's current verifier will reject the result;
- remove a platform's own disclosure duty or its independent AI detection;
- make deceptive or unlawful use permissible.

An original file may remain linked to an account or generation session in a
provider's systems even after a local copy is changed.

## Legal context

Several jurisdictions now require AI generated content to carry visible or
machine readable disclosures. Most of these rules bind providers and platforms;
some also restrict removing the labels, and China also restricts supplying
tools for doing so. The table records the primary sources checked on 2026-09-23. It is a
starting point, not a complete or current survey.

| Jurisdiction | Instrument and status | Who it binds | Removal provision |
| --- | --- | --- | --- |
| China | [Measures for Labeling AI-Generated Synthetic Content](https://www.cac.gov.cn/2025-03/14/c_1743654684782215.htm), in force since 2025-09-01; technical standard GB 45438-2025 | Service providers, platforms, app stores, users, and "any organization or individual" | Article 10: no organization or individual may maliciously delete, alter, forge, or conceal the required labels, or provide tools or services for others to do so. Article 9 lets a provider supply unlabeled output on a user's request under a user agreement, keeping logs. |
| China | [Deep Synthesis Provisions](https://www.cac.gov.cn/2022-12/11/c_1672221949354811.htm), in force since 2023-01-10 | Everyone | Article 18: no organization or individual may use technical means to delete, alter, or conceal deep synthesis labels. |
| China | CAC enforcement campaign against AI application abuses, [launched 2026-04-30](https://www.cac.gov.cn/2026-04/30/c_1779289298718765.htm) | Platforms, sellers, open-source communities | Names tutorials that teach label removal and non-compliant de-labeling tools and services as targets. |
| European Union | [AI Act Article 50](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), applying from 2026-08-02; [Regulation (EU) 2026/1744](https://eur-lex.europa.eu/eli/reg/2026/1744/oj) gives systems already on the market until 2026-12-02 | Providers (machine-readable marking) and deployers (deepfake disclosure, except purely personal non-professional use) | None in the Act. |
| European Union | [Code of Practice on marking and labelling of AI-generated content](https://digital-strategy.ec.europa.eu/en/news/commission-publishes-code-practice-marking-and-labelling-ai-generated-content), final 2026-06-10, voluntary | Signatory providers and deployers | Measure 1.2: signatories prohibit intentional removal of or tampering with metadata markings in their terms, with an exception for legitimate processing such as security audits and research purposes, and will not promote tools whose purpose is to circumvent machine-readable markings. |
| United States, California | SB 942 AI Transparency Act as amended by [AB 853](https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202520260AB853); provider duties from 2026-08-02, platforms from 2027-01-01 | Large providers, their licensees, large platforms, hosting platforms | Providers must make disclosures difficult to remove; platforms may not knowingly strip provenance data. No duty on end users or third-party tools. |
| United States, federal | [17 U.S.C. 1202](https://www.law.cornell.edu/uscode/text/17/1202) (copyright management information) | Any person | Prohibits knowingly removing copyright management information when the person knows it will enable or conceal infringement. |
| India | [IT Rules amended by G.S.R. 120(E)](https://www.meity.gov.in/static/uploads/2026/02/550681ab908f8afb135b0ad42816a1c9.pdf), notified 2026-02-10, in force since 2026-02-20; a further draft amendment (April 2026) would require the label for the whole duration of a video | Intermediaries offering tools that create synthetically generated information; significant social media intermediaries must also collect and verify user declarations | Rule 3(3)(b): the intermediary shall not enable modification, suppression, or removal of the label or the permanent metadata and unique identifier. The [MeitY FAQ](https://www.meity.gov.in/static/uploads/2025/10/065b6deb585441b5ccdf8be42502a49c.pdf) adds that intermediaries should not offer "remove watermark" or "export without metadata" functions. |
| South Korea | [AI Framework Act](https://www.law.go.kr/법령/인공지능발전과신뢰기반조성등에관한기본법), in force since 2026-01-22 | AI business operators | Labeling duty for generative output; no removal provision found. |

Provider terms of service can also restrict removal or misrepresentation of
origin independently of any statute; several major providers prohibit
presenting generated output as human-made. Platforms such as YouTube, TikTok,
and Meta require creators to disclose realistic AI content and may apply their
own labels from their own detection. A test upload on 2026-09-24 confirmed the YouTube side: a Gemini
video with Google C2PA got "How this was made: Made with AI, Info from Google
LLC" with the AI question left unanswered, while an xAI Grok video whose own
manifest was self-signed got no label; the public streams carried no C2PA,
and the owner's Studio download came back re-signed by YouTube. As of
September 2026, YouTube carries
forward C2PA 2.1 or later disclosures that the whole video was made with AI and
also labels content its own systems detect; TikTok reads C2PA Content
Credentials and adds its own invisible watermark to content made with its AI
tools; Meta labels from the C2PA and IPTC AI indicators; LinkedIn shows a C2PA
icon on signed media. Sources: [YouTube](https://support.google.com/youtube/answer/15447836),
[YouTube blog, 2026-05-27](https://blog.youtube/news-and-events/improving-ai-labels-viewers-creators/),
[TikTok, 2025-11-19](https://newsroom.tiktok.com/more-ways-to-spot-shape-and-understand-ai-content?lang=en),
[Meta](https://about.fb.com/news/2024/04/metas-approach-to-labeling-ai-generated-content-and-manipulated-media/),
[LinkedIn](https://www.linkedin.com/help/linkedin/answer/a6282984).

Before removing a mark, consider:

1. whether you own or are authorized to edit the content;
2. whether a disclosure is legally required where the content will be used;
3. whether removal would mislead a viewer about authorship or origin;
4. whether copyright management information is involved;
5. whether a platform's or provider's terms prohibit the change;
6. whether the jurisdiction restricts label removal or removal tools.

## Appropriate uses

Examples that fit the project scope include:

- testing the robustness of watermarking and provenance systems;
- evaluating image processing pipelines in a controlled environment;
- removing metadata that exposes an account identifier from your own file;
- correcting an AI label applied to a human photograph after a limited edit;
- publishing your own generated artwork under the disclosure rules that apply
  to you.

## Uses the project does not condone

- fraud or impersonation;
- nonconsensual sexual imagery;
- hiding copyright infringement;
- presenting generated content as human made where that claim is deceptive;
- evading a disclosure that the law requires;
- removing protection from someone else's paid asset.

## Reporting security or safety concerns

Open a GitHub issue when the concern can be discussed publicly without exposing
private data. Do not attach confidential images, credentials, or personal
information to a public issue.
