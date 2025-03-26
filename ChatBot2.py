
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import getpass
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

###Load in onePDF file
#def PdfFile(Pdf_doc):
#    #file_path = "PaperPdf.pdf"
#    file_path = Pdf_doc
#    loader = PyPDFLoader(file_path)
#    docs = loader.load()
#    return docs
def LoadDocs(file_path):
    if file_path is None:
        file_path = "PaperPdf_HighResolution.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

#print(len(docs))
#print(docs[5].page_content[0:10000])
#print(docs[0].metadata)
def LoadAPI(key,UserInput,docs):
    os.environ["OPENAI_API_KEY"] = key
    # "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"
    llm = ChatOpenAI(model="gpt-4o")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    persist_dir_path = "./chroma_db"
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding,persist_directory=persist_dir_path)
    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    #results = rag_chain.invoke({"input": "What is the survival rate of the HER-2 oncogene?"})
    results = rag_chain.invoke({"input": UserInput})
    return results
# print(results)
# #{'input': 'What is HER-2 oncogene?', 'context': [Document(id='2924d8d2-c059-4378-b904-0951279f7287', metadata={'page': 2, 'source': 'PaperPdf.pdf'}, page_content='somalmappingstudiesrevealedallthreegenes(neu,c-erbB-2,and\nHER-2)tobethesame(22,24,25).Afourthgroup,alsousingv-\nerbBasaprobe,identifiedthesamegeneinamammarycarcinoma\ncellline,MAC117,whereitwasfoundtobeamplifiedfive-toten-\nfold(26).\nThisgene,whichwewillcallHER-2/neu,encodesanewmember\nofthetyrosinekinasefamily;andisdoselyrelatedto,butdistinct\nfrom,theEGFRgene(22).HER-2/neudiffersfromEGFRinthatit\nisfoundonbandq21ofchromosome17(22,24,25),ascompared\ntobandpll.-p13ofchromosome 7,wheretheEGFRgeneis\nlocated(27).Also,theHER-2/neugenegeneratesamessenger\nRNA(mRNA)of4.8kb(22),whichdiffersfromthe5.8-and10-\nkbtransciptsfortheEGFRgene(28).Finally,theproteinencoded\nbytheHER-2/neugeneis185,000daltons(21),ascomparedtothe\n170,000-dalton proteinencodedbytheEGFRgene.Conversely,on\nthebasisofsequencedata,HER-2/neuismorecloselyrelatedtothe\nEGFRgenethantoothermembersofthetyrosinekinasefamily\n(22).LiketheEGFRprotem,HER-2/neuhasanextracellular\ndomain,atransmembrane domainthatincludestwocysteine-rich'), Document(id='ef6beeff-82ee-4098-b25c-ba0af76130f6', metadata={'page': 2, 'source': 'PaperPdf.pdf'}, page_content='somalmappingstudiesrevealedallthreegenes(neu,c-erbB-2,and\nHER-2)tobethesame(22,24,25).Afourthgroup,alsousingv-\nerbBasaprobe,identifiedthesamegeneinamammarycarcinoma\ncellline,MAC117,whereitwasfoundtobeamplifiedfive-toten-\nfold(26).\nThisgene,whichwewillcallHER-2/neu,encodesanewmember\nofthetyrosinekinasefamily;andisdoselyrelatedto,butdistinct\nfrom,theEGFRgene(22).HER-2/neudiffersfromEGFRinthatit\nisfoundonbandq21ofchromosome17(22,24,25),ascompared\ntobandpll.-p13ofchromosome 7,wheretheEGFRgeneis\nlocated(27).Also,theHER-2/neugenegeneratesamessenger\nRNA(mRNA)of4.8kb(22),whichdiffersfromthe5.8-and10-\nkbtransciptsfortheEGFRgene(28).Finally,theproteinencoded\nbytheHER-2/neugeneis185,000daltons(21),ascomparedtothe\n170,000-dalton proteinencodedbytheEGFRgene.Conversely,on\nthebasisofsequencedata,HER-2/neuismorecloselyrelatedtothe\nEGFRgenethantoothermembersofthetyrosinekinasefamily\n(22).LiketheEGFRprotem,HER-2/neuhasanextracellular\ndomain,atransmembrane domainthatincludestwocysteine-rich'), Document(id='5c0324aa-4423-4d4d-90e9-102e4b00655b', metadata={'page': 2, 'source': 'PaperPdf.pdf'}, page_content='HumanBreastCancer:Correlation of\nRelapse andSurvival withAmplification\noftheHER-2lneu Oncogene\nDENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,\nAxELULLRICH,WILLiAML.McGuIRE\nTheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the\nepidermalgrowthfactorreceptr.Thisgenehasbeen\nshowntobeamplifiedihumanbrtcancercelllines.\nInthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-\nfoldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-\ndictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,'), Document(id='2f1c2af5-3433-4436-836d-b00412748e6a', metadata={'page': 2, 'source': 'PaperPdf.pdf'}, page_content='HumanBreastCancer:Correlation of\nRelapse andSurvival withAmplification\noftheHER-2lneu Oncogene\nDENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,\nAxELULLRICH,WILLiAML.McGuIRE\nTheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the\nepidermalgrowthfactorreceptr.Thisgenehasbeen\nshowntobeamplifiedihumanbrtcancercelllines.\nInthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-\nfoldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-\ndictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,')], 'answer': 'The HER-2 oncogene, also known as HER-2/neu, is a member of the tyrosine kinase family related to—but distinct from—the epidermal growth factor receptor (EGFR). It is located on chromosome 17 at band q21 and encodes a 185,000-dalton protein. HER-2 is often amplified in various cancers, notably in 30% of breast cancers, and its amplification is a significant predictor of both overall survival and time to relapse.'}
# print(results["context"][0].page_content)
# for i in range(len(results["context"])):
#     print(results["context"][i].page_content)

#{'input': 'What is the survival rate of the HER-2 oncogene?', 'context': [Document(id='756a8b8e-b4e0-4f36-87df-6683bcb0e2bf', metadata={'page': 2, 'source': 'PaperPdf.pdf'}, page_content='HumanBreastCancer:Correlation of\nRelapse andSurvival withAmplification\noftheHER-2lneu Oncogene\nDENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,\nAxELULLRICH,WILLiAML.McGuIRE\nTheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the\nepidermalgrowthfactorreceptr.Thisgenehasbeen\nshowntobeamplifiedihumanbrtcancercelllines.\nInthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-\nfoldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-\ndictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,'), Document(id='5c0324aa-4423-4d4d-90e9-102e4b00655b', metadata={'page': 2, 'source': 'PaperPdf.pdf'}, page_content='HumanBreastCancer:Correlation of\nRelapse andSurvival withAmplification\noftheHER-2lneu Oncogene\nDENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,\nAxELULLRICH,WILLiAML.McGuIRE\nTheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the\nepidermalgrowthfactorreceptr.Thisgenehasbeen\nshowntobeamplifiedihumanbrtcancercelllines.\nInthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-\nfoldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-\ndictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,'), Document(id='2f1c2af5-3433-4436-836d-b00412748e6a', metadata={'page': 2, 'source': 'PaperPdf.pdf'}, page_content='HumanBreastCancer:Correlation of\nRelapse andSurvival withAmplification\noftheHER-2lneu Oncogene\nDENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,\nAxELULLRICH,WILLiAML.McGuIRE\nTheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the\nepidermalgrowthfactorreceptr.Thisgenehasbeen\nshowntobeamplifiedihumanbrtcancercelllines.\nInthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-\nfoldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-\ndictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,'), Document(id='a89153f9-0584-4bee-b0cc-c83c00ac7606', metadata={'page': 1, 'source': 'PaperPdf.pdf'}, page_content='DOI: 10.1126/science.3798106 , 177 (1987); 235Science  et al. DJ Slamon,oncogenesurvival with amplification of the HER-2/neu Human breast cancer: correlation of relapse and\n www.sciencemag.org (this information is current as of January 15, 2007 ):The following resources related to this article are available online at\n http://www.sciencemag.orgversion of this article at:  including high-resolution figures, can be found in the online Updated information and services,\n http://www.sciencemag.org#otherarticles, 15 of which can be accessed for free: cites 37 articles This article \n http://www.sciencemag.org#otherarticles 99 articles hosted by HighWire Press; see: cited by This article has been \n http://www.sciencemag.org/help/about/permissions.dtl in whole or in part can be found at: this articlepermission to reproduce  of this article or about obtaining reprints Information about obtaining')], 'answer': "The retrieved context does not provide specific survival rates for breast cancer patients with HER-2 amplification. It indicates that HER-2/neu gene amplification is a significant predictor of overall survival and time to relapse in breast cancer patients, but does not mention specific survival rates. Therefore, I don't have that information."}
#{'input': 'What is the survival rate of the HER-2 oncogene?', 'context': [Document(id='756a8b8e-b4e0-4f36-87df-6683bcb0e2bf', metadata={'page': 2, 'source': 'PaperPdf.pdf'}, page_content='HumanBreastCancer:Correlation of\nRelapse andSurvival withAmplification\noftheHER-2lneu Oncogene\nDENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,\nAxELULLRICH,WILLiAML.McGuIRE\nTheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the\nepidermalgrowthfactorreceptr.Thisgenehasbeen\nshowntobeamplifiedihumanbrtcancercelllines.\nInthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-\nfoldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-\ndictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,'), Document(id='5c0324aa-4423-4d4d-90e9-102e4b00655b', metadata={'page': 2, 'source': 'PaperPdf.pdf'}, page_content='HumanBreastCancer:Correlation of\nRelapse andSurvival withAmplification\noftheHER-2lneu Oncogene\nDENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,\nAxELULLRICH,WILLiAML.McGuIRE\nTheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the\nepidermalgrowthfactorreceptr.Thisgenehasbeen\nshowntobeamplifiedihumanbrtcancercelllines.\nInthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-\nfoldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-\ndictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,'), Document(id='2f1c2af5-3433-4436-836d-b00412748e6a', metadata={'page': 2, 'source': 'PaperPdf.pdf'}, page_content='HumanBreastCancer:Correlation of\nRelapse andSurvival withAmplification\noftheHER-2lneu Oncogene\nDENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,\nAxELULLRICH,WILLiAML.McGuIRE\nTheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the\nepidermalgrowthfactorreceptr.Thisgenehasbeen\nshowntobeamplifiedihumanbrtcancercelllines.\nInthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-\nfoldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-\ndictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,'), Document(id='74d87c99-e2c3-41ec-a391-241a7b5037e6', metadata={'page': 5, 'source': 'PaperPdf_HighResolution.pdf'}, page_content='alone, 47%; adjuvant systemic therapy plus local radiation, 19%; \nand local radiation alone, 17%. A strong and highly statistically \nsignificant correlation was found between the degree of gene \namplification and both time to disease relapse (P = <0.0001) and \nsurvival (P = 0.0011) (Table 4). Moreover, when compared in \nunivariate analyses to other parameters, amplification of HER-2/new \nwas found to be superior to all other prognostic factors, with the \nexception of the number of positive nodes (which it equaled) in \npredicting time to relapse and overall survival in human breast \ncancer (Table 4). The association between HER-2/nex amplification \nand relapse and survival can be illustrated graphically in actuarial \nsurvival curves (Fig. 3, A to D). While there was a somewhat \nshortened time to relapse and shorter overall survival in patients \nhaving any amplification of the HER-2/new gene in their tumors \n(Fig. 3, A and B), the greatest differences were found when')], 'answer': 'The specific survival rate of patients with breast cancer who have amplification of the HER-2 oncogene is not explicitly stated, but the amplification of HER-2/neu is associated with a significant predictor of overall survival and time to relapse. It is noted that there is a strong correlation between HER-2/neu amplification and a shorter time to relapse and shorter overall survival. The study indicates that HER-2/neu amplification has greater prognostic value than most other currently used prognostic factors.'}

# HumanBreastCancer:Correlation of
# Relapse andSurvival withAmplification
# oftheHER-2lneu Oncogene
# DENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,
# AxELULLRICH,WILLiAML.McGuIRE
# TheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the
# epidermalgrowthfactorreceptr.Thisgenehasbeen
# showntobeamplifiedihumanbrtcancercelllines.
# Inthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-
# foldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-
# dictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,
# HumanBreastCancer:Correlation of
# Relapse andSurvival withAmplification
# oftheHER-2lneu Oncogene
# DENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,
# AxELULLRICH,WILLiAML.McGuIRE
# TheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the
# epidermalgrowthfactorreceptr.Thisgenehasbeen
# showntobeamplifiedihumanbrtcancercelllines.
# Inthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-
# foldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-
# dictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,
# HumanBreastCancer:Correlation of
# Relapse andSurvival withAmplification
# oftheHER-2lneu Oncogene
# DENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,
# AxELULLRICH,WILLiAML.McGuIRE
# TheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the
# epidermalgrowthfactorreceptr.Thisgenehasbeen
# showntobeamplifiedihumanbrtcancercelllines.
# Inthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-
# foldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-
# dictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,
# HumanBreastCancer:Correlation of
# Relapse andSurvival withAmplification
# oftheHER-2lneu Oncogene
# DENNISJ.SLAMON,*GARYM.CLARK,STEVENG.WONG,WENDYJ.LEVIN,
# AxELULLRICH,WILLiAML.McGuIRE
# TheHER-2/neuoncoFeneisamemberoftheerbB-likeoncogenefamily,andaSrelatedto,butdistinctfirom,the
# epidermalgrowthfactorreceptr.Thisgenehasbeen
# showntobeamplifiedihumanbrtcancercelllines.
# Inthecurrentstudy,alterationsofthegenein189primaryhumanbreastcancerswereinstigatedHER-2/neuwasfoundtobeamplifiedfrm2-toeaterthan20-
# foldin30%ofthetumors.Correlationofgeneamplifica-tionwithseveraldiseaseparameterswasevaluatedAm-plificationoftheHER-2/neugenewasasignificantpre-
# dictorofbothoverallsurvivalandtimetorelapseinpatientswithbreastcancer.Itretaineditssignificanceevenwhenadjustmentsweremadeforotherknownprognosticfactors.Moreover,HER-2/neuamplificationhadgreaterprognosticvaluethanmostcurrentlyusedprognosticfactors,incudinghormonal-receptor status,
# alone, 47%; adjuvant systemic therapy plus local radiation, 19%;
# and local radiation alone, 17%. A strong and highly statistically
# significant correlation was found between the degree of gene
# amplification and both time to disease relapse (P = <0.0001) and
# survival (P = 0.0011) (Table 4). Moreover, when compared in
# univariate analyses to other parameters, amplification of HER-2/new
# was found to be superior to all other prognostic factors, with the
# exception of the number of positive nodes (which it equaled) in
# predicting time to relapse and overall survival in human breast
# cancer (Table 4). The association between HER-2/nex amplification
# and relapse and survival can be illustrated graphically in actuarial
# survival curves (Fig. 3, A to D). While there was a somewhat
# shortened time to relapse and shorter overall survival in patients
# having any amplification of the HER-2/new gene in their tumors
# (Fig. 3, A and B), the greatest differences were found when
import sys
def main():
    # Read from stdin
    print("Welcome to the initial trial of the chatbot!")
    user_input1 = input("Please enter the pdf path, e.g. PaperPdf_HighResolution.pdf  ")
    docs= LoadDocs(user_input1)
    key = input("Please enter your openAI API key:  ")
    # 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
    UserInput = input("What would you like to ask regarding to the doc you uploaded:")
    results = LoadAPI(key, UserInput,docs)
    #"What is the survival rate of the HER-2 oncogene?"
    for i in range(len(results["context"])):
        print("--------------------------------------------------------")
        print("The results for retrieval " + str(i) +" is: ")
        print(results["context"][i].page_content)

if __name__ == "__main__":
    main()



