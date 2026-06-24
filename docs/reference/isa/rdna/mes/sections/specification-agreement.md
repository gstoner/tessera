# Specification Agreement

> Micro Engine Scheduler ISA — pages 2–4

Specification Agreement
             This Specification Agreement (this "Agreement") is a legal agreement between Advanced Micro
             Devices, Inc. ("AMD") and "You" as the recipient of the attached AMD Specification (the
             "Specification"). If you are accessing the Specification as part of your performance of work for
             another party, you acknowledge that you have authority to bind such party to the terms and
             conditions of this Agreement. If you accessed the Specification by any means or otherwise use
             or provide Feedback (defined below) on the Specification, You agree to the terms and conditions
             set forth in this Agreement. If You do not agree to the terms and conditions set forth in this
             Agreement, you are not licensed to use the Specification; do not use, access or provide
             Feedback about the Specification. In consideration of Your use or access of the Specification (in
             whole or in part), the receipt and sufficiency of which are acknowledged, You agree as follows:

                1. You may review the Specification only (a) as a reference to assist You in planning and
                   designing Your product, service or technology ("Product") to interface with an AMD
                   product in compliance with the requirements as set forth in the Specification and (b) to
                   provide Feedback about the information disclosed in the Specification to AMD.

                2. Except as expressly set forth in Paragraph 1, all rights in and to the Specification are
                   retained by AMD. This Agreement does not give You any rights under any AMD patents,
                   copyrights, trademarks or other intellectual property rights. You may not (i) duplicate any
                   part of the Specification; (ii) remove this Agreement or any notices from the
                   Specification, or (iii) give any part of the Specification, or assign or otherwise provide
                   Your rights under this Agreement, to anyone else.

                3. The Specification may contain preliminary information, errors, or inaccuracies, or may not
                   include certain necessary information. Additionally, AMD reserves the right to
                   discontinue or make changes to the Specification and its products at any time without
                   notice. The Specification is provided entirely "AS IS." AMD MAKES NO WARRANTY OF
                   ANY KIND AND DISCLAIMS ALL EXPRESS, IMPLIED AND STATUTORY WARRANTIES,
                   INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY,
                   FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, TITLE OR THOSE
                   WARRANTIES ARISING AS A COURSE OF DEALING OR CUSTOM OF TRADE. AMD
                   SHALL NOT BE LIABLE FOR DIRECT, INDIRECT, CONSEQUENTIAL, SPECIAL,
                   INCIDENTAL, PUNITIVE OR EXEMPLARY DAMAGES OF ANY KIND (INCLUDING LOSS
                   OF BUSINESS, LOSS OF INFORMATION OR DATA, LOST PROFITS, LOSS OF CAPITAL,
                   LOSS OF GOODWILL) REGARDLESS OF THE FORM OF ACTION WHETHER IN
                   CONTRACT, TORT (INCLUDING NEGLIGENCE) AND STRICT PRODUCT LIABILITY OR
                   OTHERWISE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

                4. Furthermore, AMD’s products are not designed, intended, authorized or warranted for
                   use as components in systems intended for surgical implant into the body, or in other
                   applications intended to support or sustain life, or in any other application in which the
                   failure of AMD’s product could create a situation where personal injury, death, or severe
                   property or environmental damage may occur.

                5. You have no obligation to give AMD any suggestions, comments or feedback
                   ("Feedback") relating to the Specification. However, any Feedback You voluntarily

April 2024                                                                                               2

                provide may be used by AMD without restriction, fee or obligation of confidentiality.
                Accordingly, if You do give AMD Feedback on any version of the Specification, You
                agree AMD may freely use, reproduce, license, distribute, and otherwise commercialize
                Your Feedback in any product, as well as has the right to sublicense third parties to do
                the same. Further, You will not give AMD any Feedback that You may have reason to
                believe is (i) subject to any patent, copyright or other intellectual property claim or right
                of any third party; or (ii) subject to license terms which seek to require any product or
                intellectual property incorporating or derived from Feedback or any Product or other
                AMD intellectual property to be licensed to or otherwise provided to any third party.

             6. You shall adhere to all applicable U.S., European, and other export laws, including but not
                limited to the U.S. Export Administration Regulations ("EAR"), (15 C.F.R. Sections 730
                through 774), and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009. Further,
                pursuant to Section 740.6 of the EAR, You hereby certifies that, except pursuant to a
                license granted by the United States Department of Commerce Bureau of Industry and
                Security or as otherwise permitted pursuant to a License Exception under the U.S. Export
                Administration Regulations ("EAR"), You will not (1) export, re-export or release to a
                national of a country in Country Groups D:1, E:1 or E:2 any restricted technology,
                software, or source code You receive hereunder, or (2) export to Country Groups D:1,
                E:1 or E:2 the direct product of such technology or software, if such foreign produced
                direct product is subject to national security controls as identified on the Commerce
                Control List (currently found in Supplement 1 to Part 774 of EAR). For the most current
                Country Group listings, or for additional information about the EAR or Your obligations
                under those regulations, please refer to the U.S. Bureau of Industry and Security’s
                website at http://www.bis.doc.gov/.

             7. If You are a part of the U.S. Government, then the Specification is provided with
                "RESTRICTED RIGHTS" as set forth in subparagraphs (c) (1) and (2) of the Commercial
                Computer Software-Restricted Rights clause at FAR 52.227-14 or subparagraph (c) (1)(ii)
                of the Rights in Technical Data and Computer Software clause at DFARS 252.277-7013,
                as applicable.

             8. This Agreement is governed by the laws of the State of California without regard to its
                choice of law principles. Any dispute involving it must be brought in a court having
                jurisdiction of such dispute in Santa Clara County, California, and You waive any
                defenses and rights allowing the dispute to be litigated elsewhere. If any part of this
                agreement is unenforceable, it will be considered modified to the extent necessary to
                make it enforceable, and the remainder shall continue in effect. The failure of AMD to
                enforce any rights granted hereunder or to take action against You in the event of any
                breach hereunder shall not be deemed a waiver by AMD as to subsequent enforcement
                of rights or subsequent actions in the event of future breaches. This Agreement is the
                entire agreement between You and AMD concerning the Specification; it may be
                changed only by a written document signed by both You and an authorized
                representative of AMD.

April 2024                                                                                               3

Contents
MICRO ENGINE SCHEDULER SPECIFICATION ...................................................................................................................................... 1
  SPECIFICATION AGREEMENT ............................................................................................................................................................................... 2
  SCHEDULING REQUIREMENTS .............................................................................................................................................................................. 5
  HW ARCHITECTURE OVERVIEW............................................................................................................................................................................ 6
  SCHEDULER FW ARCHITECTURE ........................................................................................................................................................................... 8
  SCHEDULING ALGORITHM ................................................................................................................................................................................. 10
    Queue state transitions .......................................................................................................................................................................... 11
    Round robin scheduling .......................................................................................................................................................................... 12
    Queue prioritization ............................................................................................................................................................................... 13
  MES API ...................................................................................................................................................................................................... 16
    MES API format ...................................................................................................................................................................................... 16
    MES_SCH_API_SET_HW_RSRC ............................................................................................................................................................... 18
    MES_SCH_API_ADD_QUEUE .................................................................................................................................................................. 22
    MES_SCH_API_AMD_LOG ...................................................................................................................................................................... 26
    MES_SCH_API_REMOVE_QUEUE ........................................................................................................................................................... 26
    MES_SCH_API_SET_SCHEDULING_CONFIG ........................................................................................................................................... 27
    MES_SCH_API_PERFORM_YIELD............................................................................................................................................................ 28
    MES_SCH_API_SET_GANG_PRIORITY_LEVEL ......................................................................................................................................... 29
    MES_SCH_API_SUSPEND........................................................................................................................................................................ 29
    MES_SCH_API_RESUME ......................................................................................................................................................................... 30
    MES_SCH_API_RESET ............................................................................................................................................................................. 31
    MES_SCH_API_SET_LOG_BUFFER .......................................................................................................................................................... 34
    MES_SCH_API_CHANGE_GANG_PRORITY ............................................................................................................................................. 34
    MES_SCH_API_QUERY_SCHEDULER_STATUS ........................................................................................................................................ 36
    MES_API_QUERY_MES__GET_CTX_ARRAY_SIZE ................................................................................................................................... 36
    MES_API_QUERY_MES__HEALTHY_CHECK ........................................................................................................................................... 37
    MES_SCH_API_PROGRAM_GDS ............................................................................................................................................................. 37
    MES_SCH_API_SET_DEBUG_VMID ........................................................................................................................................................ 38
    MES_SCH_API_UPDATE_ROOT_PAGE_TABLE ....................................................................................................................................... 40
    MES_SCH_API_SET_SE_MODE ............................................................................................................................................................... 40
    MES_SCH_API_SET_GANG_SUBMIT ...................................................................................................................................................... 41
    MES_SCH_API_MISC .............................................................................................................................................................................. 42
    MESAPI_MISC__WRITE_REG.................................................................................................................................................................. 43
    MESAPI_MISC__INV_GART .................................................................................................................................................................... 43
    MESAPI_MISC__QUERY_STATUS ........................................................................................................................................................... 44
    MESAPI_MISC__READ_REG ................................................................................................................................................................... 44
    MESAPI_MISC__WAIT_REG_MEM ......................................................................................................................................................... 44
    MESAPI_MISC__SET_SHADER_DEBUGGER ............................................................................................................................................ 45
    MESAPI_MISC__NOTIFY_WORK_ON_UNMAPPED_QUEUE................................................................................................................... 46
    MESAPI_MISC__NOTIFY_TO_UNMAP_PROCESSES ............................................................................................................................... 46
  SCHEDULER LOG ............................................................................................................................................................................................. 47
    API history .............................................................................................................................................................................................. 47
    Event log history ..................................................................................................................................................................................... 48
    Interrupt history ..................................................................................................................................................................................... 49
    Example of log usage ............................................................................................................................................................................. 50
  DISCLAIMER ................................................................................................................................................................................................... 52
  NOTICES........................................................................................................................................................................................................ 53
  TRADEMARKS ................................................................................................................................................................................................. 54

 April 2024                                                                                                                                                                                 4
