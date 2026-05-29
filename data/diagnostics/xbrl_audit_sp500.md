# XBRL Label Audit

**Sample:** 50 S&P 500 companies (sampled from Wikipedia constituent list)  
**Seed:** 42  
**Companies sampled:** Nasdaq, Inc., BALL CORPORATION, AKAMAI TECHNOLOGIES, INC., PTC Inc., DANAHER CORP /DE/, CONSTELLATION BRANDS, INC., CMS ENERGY CORP, BOSTON SCIENTIFIC CORP, Prudential Financial, Inc., AUTOZONE INC, O Reilly Automotive Inc, UDR, Inc., Kraft Heinz Co, ARES MANAGEMENT CORPORATION, Masco Corporation, GENERAC HOLDINGS INC., Allegion plc, ALIGN TECHNOLOGY, INC., Assurant, Inc., CITIZENS FINANCIAL GROUP INC/RI, COLGATE-PALMOLIVE COMPANY, Intuitive Surgical, Inc., Merck & Co., Inc., Albemarle Corporation, LENNOX INTERNATIONAL INC, Chevron Corp, Philip Morris International Inc., NEWS CORPORATION, PARKER-HANNIFIN CORPORATION, GE Vernova Inc., CLOROX CO /DE/, HCA Healthcare, Inc., MARTIN MARIETTA MATERIALS INC, Datadog, Inc., Smurfit Westrock plc, TRADE DESK, INC., AbbVie Inc., RTX CORPORATION, SKYWORKS SOLUTIONS, INC., CADENCE DESIGN SYSTEMS, INC., Palo Alto Networks Inc, EOG RESOURCES, INC., BXP, INC., Citigroup Inc, WESTERN DIGITAL CORPORATION, REGENCY CENTERS CORPORATION, EMERSON ELECTRIC CO., Fair Isaac Corp, Atmos Energy Corp, EVEREST GROUP, LTD.

---

## Summary

| Concept | Distinct labels | No-match companies |
|---------|----------------|-------------------|
| Revenue | 12 | ✓ 0/50 |
| Gross Profit | 3 | ⚠ 17/50 |
| Net Income | 39 | ✓ 0/50 |
| Operating Cash Flow | 12 | ✓ 0/50 |
| CapEx | 2 | ⚠ 4/50 |
| Total Debt | 82 | ⚠ 2/50 |
| Cash & Equivalents | 14 | ✓ 0/50 |
| Stockholders Equity | 10 | ✓ 0/50 |
| Interest Expense | 30 | ⚠ 2/50 |
| Depreciation | 53 | ✓ 0/50 |
| Tax Expense | 20 | ⚠ 1/50 |

---

## Revenue

| Label | n/50 | Coverage |
|-------|------|---------|
| `RevenueFromContractWithCustomerExcludingAssessedTax` | 36 | 72% ███████░░░ |
| `Revenues` | 35 | 70% ███████░░░ |
| `SalesRevenueNet` | 27 | 54% █████░░░░░ |
| `SalesRevenueGoodsNet` | 13 | 26% ███░░░░░░░ |
| `RevenueFromContractWithCustomerIncludingAssessedTax` | 10 | 20% ██░░░░░░░░ |
| `SalesRevenueServicesNet` | 7 | 14% █░░░░░░░░░ |
| `OtherSalesRevenueNet` | 4 | 8% █░░░░░░░░░ |
| `RevenuesFromTransactionsWithOtherOperatingSegmentsOfSameEntity` | 3 | 6% █░░░░░░░░░ |
| `RelatedPartyTransactionOtherRevenuesFromTransactionsWithRelatedParty` | 2 | 4% ░░░░░░░░░░ |
| `SalesRevenueServicesGross` | 1 | 2% ░░░░░░░░░░ |
| `ResultsOfOperationsSalesRevenueToUnaffiliatedEnterprises` | 1 | 2% ░░░░░░░░░░ |
| `RevenuesFromExternalCustomers` | 1 | 2% ░░░░░░░░░░ |

**✓ All 50 companies have at least one matching label.**

---

## Gross Profit

| Label | n/50 | Coverage |
|-------|------|---------|
| `GrossProfit` | 32 | 64% ██████░░░░ |
| `DisposalGroupIncludingDiscontinuedOperationGrossProfitLoss` | 4 | 8% █░░░░░░░░░ |
| `EquityMethodInvestmentSummarizedFinancialInformationGrossProfitLoss` | 2 | 4% ░░░░░░░░░░ |

**⚠ No match in 17/50 companies:** *AKAMAI TECHNOLOGIES, INC.*, *CMS ENERGY CORP*, *Prudential Financial, Inc.*, *UDR, Inc.*, *ARES MANAGEMENT CORPORATION*, *Allegion plc*, *Assurant, Inc.*, *CITIZENS FINANCIAL GROUP INC/RI*, *Merck & Co., Inc.*, *Chevron Corp*, *NEWS CORPORATION*, *HCA Healthcare, Inc.*, *TRADE DESK, INC.*, *CADENCE DESIGN SYSTEMS, INC.*, *EOG RESOURCES, INC.*, *REGENCY CENTERS CORPORATION*, *EVEREST GROUP, LTD.*

---

## Net Income

| Label | n/50 | Coverage |
|-------|------|---------|
| `NetIncomeLoss` | 50 | 100% ██████████ |
| `ProfitLoss` | 40 | 80% ████████░░ |
| `NetIncomeLossAttributableToNoncontrollingInterest` | 34 | 68% ███████░░░ |
| `NetIncomeLossAvailableToCommonStockholdersBasic` | 32 | 64% ██████░░░░ |
| `NetIncomeLossAvailableToCommonStockholdersDiluted` | 27 | 54% █████░░░░░ |
| `BusinessAcquisitionsProFormaNetIncomeLoss` | 20 | 40% ████░░░░░░ |
| `AdjustmentsNoncashItemsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivitiesOther` | 17 | 34% ███░░░░░░░ |
| `EquityMethodInvestmentSummarizedFinancialInformationNetIncomeLoss` | 8 | 16% ██░░░░░░░░ |
| `NetIncomeLossFromContinuingOperationsAvailableToCommonShareholdersBasic` | 7 | 14% █░░░░░░░░░ |
| `AdjustmentsNoncashItemsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities` | 6 | 12% █░░░░░░░░░ |
| `NetIncomeLossAttributableToRedeemableNoncontrollingInterest` | 6 | 12% █░░░░░░░░░ |
| `NetIncomeLossFromContinuingOperationsAvailableToCommonShareholdersDiluted` | 5 | 10% █░░░░░░░░░ |
| `NetIncomeLossFromDiscontinuedOperationsAvailableToCommonShareholdersBasic` | 4 | 8% █░░░░░░░░░ |
| `DisposalGroupIncludingDiscontinuedOperationGrossProfitLoss` | 4 | 8% █░░░░░░░░░ |
| `BusinessAcquisitionProFormaNetIncomeLoss` | 3 | 6% █░░░░░░░░░ |
| `AdjustmentsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities` | 3 | 6% █░░░░░░░░░ |
| `MinorityInterestInNetIncomeLossOperatingPartnerships` | 3 | 6% █░░░░░░░░░ |
| `EquityMethodInvestmentSummarizedFinancialInformationGrossProfitLoss` | 2 | 4% ░░░░░░░░░░ |
| `NetIncomeLossIncludingPortionAttributableToNonredeemableNoncontrollingInterest` | 2 | 4% ░░░░░░░░░░ |
| `SegmentReportingReconcilingItemForOperatingProfitLossFromSegmentToConsolidatedAmount` | 2 | 4% ░░░░░░░░░░ |
| `SegmentReportingSegmentOperatingProfitLoss` | 2 | 4% ░░░░░░░░░░ |
| `NoncontrollingInterestInNetIncomeLossOperatingPartnershipsRedeemable` | 2 | 4% ░░░░░░░░░░ |
| `NoncontrollingInterestInNetIncomeLossLimitedPartnershipsNonredeemable` | 2 | 4% ░░░░░░░░░░ |
| `NetIncomeLossFromDiscontinuedOperationsAvailableToCommonShareholdersDiluted` | 2 | 4% ░░░░░░░░░░ |
| `SegmentReportingReconcilingItemsForOperatingProfitLoss` | 1 | 2% ░░░░░░░░░░ |
| `MinorityInterestInNetIncomeLossOtherMinorityInterests` | 1 | 2% ░░░░░░░░░░ |
| `NoncontrollingInterestInNetIncomeLossOtherNoncontrollingInterestsNonredeemable` | 1 | 2% ░░░░░░░░░░ |
| `NetIncomeLossAttributableToNonredeemableNoncontrollingInterest` | 1 | 2% ░░░░░░░░░░ |
| `NetIncomeLossPerOutstandingLimitedPartnershipUnitBasicNetOfTax` | 1 | 2% ░░░░░░░░░░ |
| `NetIncomeLossNetOfTaxPerOutstandingLimitedPartnershipUnitDiluted` | 1 | 2% ░░░░░░░░░░ |
| `SegmentReportingInformationProfitLoss` | 1 | 2% ░░░░░░░░░░ |
| `MinorityInterestInNetIncomeLossOfConsolidatedEntities` | 1 | 2% ░░░░░░░░░░ |
| `MinorityInterestInNetIncomeLossLimitedPartnerships` | 1 | 2% ░░░░░░░░░░ |
| `MinorityInterestInNetIncomeLossPreferredUnitHolders` | 1 | 2% ░░░░░░░░░░ |
| `NoncontrollingInterestInNetIncomeLossPreferredUnitHoldersRedeemable` | 1 | 2% ░░░░░░░░░░ |
| `SalesTypeAndDirectFinancingLeasesProfitLoss` | 1 | 2% ░░░░░░░░░░ |
| `NoncontrollingInterestInNetIncomeLossJointVenturePartnersNonredeemable` | 1 | 2% ░░░░░░░░░░ |
| `NetIncomeLossAllocatedToGeneralPartners` | 1 | 2% ░░░░░░░░░░ |
| `NoncontrollingInterestInNetIncomeLossOperatingPartnershipsNonredeemable` | 1 | 2% ░░░░░░░░░░ |

**✓ All 50 companies have at least one matching label.**

---

## Operating Cash Flow

| Label | n/50 | Coverage |
|-------|------|---------|
| `NetCashProvidedByUsedInOperatingActivities` | 50 | 100% ██████████ |
| `NetCashProvidedByUsedInOperatingActivitiesContinuingOperations` | 24 | 48% █████░░░░░ |
| `ExcessTaxBenefitFromShareBasedCompensationOperatingActivities` | 21 | 42% ████░░░░░░ |
| `AdjustmentsNoncashItemsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivitiesOther` | 17 | 34% ███░░░░░░░ |
| `CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations` | 13 | 26% ███░░░░░░░ |
| `OtherOperatingActivitiesCashFlowStatement` | 12 | 24% ██░░░░░░░░ |
| `AdjustmentsNoncashItemsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities` | 6 | 12% █░░░░░░░░░ |
| `AdjustmentsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities` | 3 | 6% █░░░░░░░░░ |
| `PaymentForContingentConsiderationLiabilityOperatingActivities` | 2 | 4% ░░░░░░░░░░ |
| `PaymentsForOtherOperatingActivities` | 1 | 2% ░░░░░░░░░░ |
| `ProceedsFromInsuranceSettlementOperatingActivities` | 1 | 2% ░░░░░░░░░░ |
| `IncreaseDecreaseInRestrictedCashForOperatingActivities` | 1 | 2% ░░░░░░░░░░ |

**✓ All 50 companies have at least one matching label.**

---

## CapEx

| Label | n/50 | Coverage |
|-------|------|---------|
| `PaymentsToAcquirePropertyPlantAndEquipment` | 40 | 80% ████████░░ |
| `CapitalExpendituresIncurredButNotYetPaid` | 19 | 38% ████░░░░░░ |

**⚠ No match in 4/50 companies:** *Prudential Financial, Inc.*, *Chevron Corp*, *GE Vernova Inc.*, *EVEREST GROUP, LTD.*

---

## Total Debt

| Label | n/50 | Coverage |
|-------|------|---------|
| `LongTermDebt` | 39 | 78% ████████░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive` | 36 | 72% ███████░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo` | 36 | 72% ███████░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree` | 36 | 72% ███████░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour` | 36 | 72% ███████░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths` | 34 | 68% ███████░░░ |
| `ProceedsFromIssuanceOfLongTermDebt` | 31 | 62% ██████░░░░ |
| `LongTermDebtCurrent` | 29 | 58% ██████░░░░ |
| `RepaymentsOfLongTermDebt` | 29 | 58% ██████░░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive` | 28 | 56% ██████░░░░ |
| `LongTermDebtNoncurrent` | 26 | 52% █████░░░░░ |
| `LongTermDebtFairValue` | 21 | 42% ████░░░░░░ |
| `ShortTermBorrowings` | 19 | 38% ████░░░░░░ |
| `DebtCurrent` | 17 | 34% ███░░░░░░░ |
| `LongTermDebtAndCapitalLeaseObligations` | 17 | 34% ███░░░░░░░ |
| `ProceedsFromIssuanceOfSeniorLongTermDebt` | 14 | 28% ███░░░░░░░ |
| `LongTermDebtAndCapitalLeaseObligationsCurrent` | 12 | 24% ██░░░░░░░░ |
| `LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities` | 9 | 18% ██░░░░░░░░ |
| `OtherLongTermDebt` | 8 | 16% ██░░░░░░░░ |
| `OtherShortTermBorrowings` | 8 | 16% ██░░░░░░░░ |
| `RepaymentsOfLongTermDebtAndCapitalSecurities` | 8 | 16% ██░░░░░░░░ |
| `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedNoncurrentLiabilitiesLongTermDebt` | 7 | 14% █░░░░░░░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalRemainderOfFiscalYear` | 7 | 14% █░░░░░░░░░ |
| `ProceedsFromNotesPayable` | 6 | 12% █░░░░░░░░░ |
| `RepaymentsOfNotesPayable` | 6 | 12% █░░░░░░░░░ |
| `RepaymentsOfOtherLongTermDebt` | 6 | 12% █░░░░░░░░░ |
| `LongtermDebtWeightedAverageInterestRate` | 6 | 12% █░░░░░░░░░ |
| `LiabilitiesOtherThanLongtermDebtNoncurrent` | 6 | 12% █░░░░░░░░░ |
| `ProceedsFromIssuanceOfOtherLongTermDebt` | 5 | 10% █░░░░░░░░░ |
| `ProceedsFromRepaymentsOfLongTermDebtAndCapitalSecurities` | 4 | 8% █░░░░░░░░░ |
| `ConvertibleNotesPayableCurrent` | 4 | 8% █░░░░░░░░░ |
| `ConvertibleLongTermNotesPayable` | 4 | 8% █░░░░░░░░░ |
| `BusinessAcquisitionPurchasePriceAllocationNotesPayableAndLongTermDebt` | 4 | 8% █░░░░░░░░░ |
| `ProceedsFromRepaymentsOfNotesPayable` | 4 | 8% █░░░░░░░░░ |
| `ShortTermBankLoansAndNotesPayable` | 4 | 8% █░░░░░░░░░ |
| `ProceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet` | 4 | 8% █░░░░░░░░░ |
| `BusinessAcquisitionPurchasePriceAllocationNoncurrentLiabilitiesLongTermDebt` | 4 | 8% █░░░░░░░░░ |
| `InterestExpenseLongTermDebt` | 4 | 8% █░░░░░░░░░ |
| `ConvertibleNotesPayable` | 3 | 6% █░░░░░░░░░ |
| `OtherLongTermDebtCurrent` | 3 | 6% █░░░░░░░░░ |
| `NotesPayableCurrent` | 3 | 6% █░░░░░░░░░ |
| `UnsecuredLongTermDebt` | 3 | 6% █░░░░░░░░░ |
| `NotesPayable` | 3 | 6% █░░░░░░░░░ |
| `OtherLongTermDebtNoncurrent` | 2 | 4% ░░░░░░░░░░ |
| `LongTermDebtAndCapitalLeaseObligationsRepaymentsOfPrincipalInNextTwelveMonths` | 2 | 4% ░░░░░░░░░░ |
| `LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalAfterYearFive` | 2 | 4% ░░░░░░░░░░ |
| `LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalInYearFive` | 2 | 4% ░░░░░░░░░░ |
| `LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalInYearFour` | 2 | 4% ░░░░░░░░░░ |
| `LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalInYearTwo` | 2 | 4% ░░░░░░░░░░ |
| `LongTermDebtAndCapitalLeaseObligationsMaturitiesRepaymentsOfPrincipalInYearThree` | 2 | 4% ░░░░░░░░░░ |
| `LongTermNotesPayable` | 2 | 4% ░░░░░░░░░░ |
| `NotesPayableFairValueDisclosure` | 1 | 2% ░░░░░░░░░░ |
| `NotesPayableToBankCurrent` | 1 | 2% ░░░░░░░░░░ |
| `NotesPayableToBank` | 1 | 2% ░░░░░░░░░░ |
| `NotesPayableToBankNoncurrent` | 1 | 2% ░░░░░░░░░░ |
| `IncreaseDecreaseInNotesPayableCurrent` | 1 | 2% ░░░░░░░░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInNextRollingTwelveMonths` | 1 | 2% ░░░░░░░░░░ |
| `ProceedsFromRepaymentsOfOtherLongTermDebt` | 1 | 2% ░░░░░░░░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInRollingYearFour` | 1 | 2% ░░░░░░░░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInRollingYearThree` | 1 | 2% ░░░░░░░░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInRollingYearFive` | 1 | 2% ░░░░░░░░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInRollingAfterYearFive` | 1 | 2% ░░░░░░░░░░ |
| `LongTermDebtMaturitiesRepaymentsOfPrincipalInRollingYearTwo` | 1 | 2% ░░░░░░░░░░ |
| `SecuredDebtCurrent` | 1 | 2% ░░░░░░░░░░ |
| `FairValueOptionAggregateDifferencesLongTermDebtInstruments` | 1 | 2% ░░░░░░░░░░ |
| `LongTermDebtPercentageBearingFixedInterestRate` | 1 | 2% ░░░░░░░░░░ |
| `InterestExpenseShortTermBorrowings` | 1 | 2% ░░░░░░░░░░ |
| `InterestExpenseOtherShortTermBorrowings` | 1 | 2% ░░░░░░░░░░ |
| `LongTermDebtPercentageBearingVariableInterestRate` | 1 | 2% ░░░░░░░░░░ |
| `LongtermDebtPercentageBearingFixedInterestAmount` | 1 | 2% ░░░░░░░░░░ |
| `SecuredLongTermDebt` | 1 | 2% ░░░░░░░░░░ |
| `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedCurrentLiabilitiesLongTermDebt` | 1 | 2% ░░░░░░░░░░ |
| `BusinessAcquisitionPurchasePriceAllocationCurrentLiabilitiesLongTermDebt` | 1 | 2% ░░░░░░░░░░ |
| `ShortTermNonBankLoansAndNotesPayable` | 1 | 2% ░░░░░░░░░░ |
| `UnsecuredDebtCurrent` | 1 | 2% ░░░░░░░░░░ |
| `ConvertibleDebtCurrent` | 1 | 2% ░░░░░░░░░░ |
| `ProceedsFromSecuredNotesPayable` | 1 | 2% ░░░░░░░░░░ |
| `NotesPayableRelatedPartiesNoncurrent` | 1 | 2% ░░░░░░░░░░ |
| `OtherNotesPayable` | 1 | 2% ░░░░░░░░░░ |
| `ProceedsFromIssuanceOfSubordinatedLongTermDebt` | 1 | 2% ░░░░░░░░░░ |
| `InterestExpenseShortTermBorrowingsExcludingFederalFundsAndSecuritiesSoldUnderAgreementsToRepurchase` | 1 | 2% ░░░░░░░░░░ |
| `CapitalizationLongtermDebtAndEquity` | 1 | 2% ░░░░░░░░░░ |

**⚠ No match in 2/50 companies:** *ALIGN TECHNOLOGY, INC.*, *TRADE DESK, INC.*

---

## Cash & Equivalents

| Label | n/50 | Coverage |
|-------|------|---------|
| `CashAndCashEquivalentsAtCarryingValue` | 49 | 98% ██████████ |
| `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect` | 48 | 96% ██████████ |
| `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents` | 46 | 92% █████████░ |
| `EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents` | 35 | 70% ███████░░░ |
| `RestrictedCashAndCashEquivalentsAtCarryingValue` | 15 | 30% ███░░░░░░░ |
| `EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations` | 12 | 24% ██░░░░░░░░ |
| `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations` | 9 | 18% ██░░░░░░░░ |
| `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseExcludingExchangeRateEffect` | 8 | 16% ██░░░░░░░░ |
| `CashCashEquivalentsAndShortTermInvestments` | 4 | 8% █░░░░░░░░░ |
| `EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsDisposalGroupIncludingDiscontinuedOperations` | 3 | 6% █░░░░░░░░░ |
| `CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations` | 2 | 4% ░░░░░░░░░░ |
| `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsDisposalGroupIncludingDiscontinuedOperations` | 2 | 4% ░░░░░░░░░░ |
| `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffectDisposalGroupIncludingDiscontinuedOperations` | 1 | 2% ░░░░░░░░░░ |
| `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffectContinuingOperations` | 1 | 2% ░░░░░░░░░░ |

**✓ All 50 companies have at least one matching label.**

---

## Stockholders Equity

| Label | n/50 | Coverage |
|-------|------|---------|
| `LiabilitiesAndStockholdersEquity` | 50 | 100% ██████████ |
| `StockholdersEquity` | 50 | 100% ██████████ |
| `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` | 38 | 76% ████████░░ |
| `StockholdersEquityOther` | 19 | 38% ████░░░░░░ |
| `StockholdersEquityNoteStockSplitConversionRatio1` | 10 | 20% ██░░░░░░░░ |
| `StockholdersEquityNoteStockSplitConversionRatio` | 5 | 10% █░░░░░░░░░ |
| `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterestAdjustedBalance1` | 5 | 10% █░░░░░░░░░ |
| `StockholdersEquityNoteSpinoffTransaction` | 5 | 10% █░░░░░░░░░ |
| `StockholdersEquityBeforeTreasuryStock` | 4 | 8% █░░░░░░░░░ |
| `CommonStockholdersEquity` | 1 | 2% ░░░░░░░░░░ |

**✓ All 50 companies have at least one matching label.**

---

## Interest Expense

| Label | n/50 | Coverage |
|-------|------|---------|
| `InterestExpense` | 42 | 84% ████████░░ |
| `UnrecognizedTaxBenefitsIncomeTaxPenaltiesAndInterestExpense` | 23 | 46% █████░░░░░ |
| `InterestExpenseNonoperating` | 21 | 42% ████░░░░░░ |
| `FinanceLeaseInterestExpense` | 19 | 38% ████░░░░░░ |
| `InterestExpenseDebt` | 17 | 34% ███░░░░░░░ |
| `IncomeTaxExaminationPenaltiesAndInterestExpense` | 13 | 26% ███░░░░░░░ |
| `InterestExpenseRelatedParty` | 5 | 10% █░░░░░░░░░ |
| `InterestExpenseOther` | 4 | 8% █░░░░░░░░░ |
| `InterestExpenseLongTermDebt` | 4 | 8% █░░░░░░░░░ |
| `DebtInstrumentConvertibleInterestExpense` | 3 | 6% █░░░░░░░░░ |
| `InterestExpenseDebtExcludingAmortization` | 3 | 6% █░░░░░░░░░ |
| `DisposalGroupIncludingDiscontinuedOperationInterestExpense` | 3 | 6% █░░░░░░░░░ |
| `CashFlowHedgeGainLossReclassifiedToInterestExpenseNet` | 3 | 6% █░░░░░░░░░ |
| `InterestExpenseSubordinatedNotesAndDebentures` | 2 | 4% ░░░░░░░░░░ |
| `IncomeTaxExaminationInterestExpense` | 2 | 4% ░░░░░░░░░░ |
| `InterestExpenseDeposits` | 2 | 4% ░░░░░░░░░░ |
| `NoninterestExpense` | 2 | 4% ░░░░░░░░░░ |
| `InterestExpenseOperating` | 2 | 4% ░░░░░░░░░░ |
| `InterestExpenseFederalFundsPurchasedAndSecuritiesSoldUnderAgreementsToRepurchase` | 2 | 4% ░░░░░░░░░░ |
| `OtherNoninterestExpense` | 2 | 4% ░░░░░░░░░░ |
| `CashFlowHedgeGainReclassifiedToInterestExpense` | 1 | 2% ░░░░░░░░░░ |
| `CashFlowHedgeLossReclassifiedToInterestExpense` | 1 | 2% ░░░░░░░░░░ |
| `NoninterestExpenseRelatedToPerformanceFees` | 1 | 2% ░░░░░░░░░░ |
| `NoninterestExpenseInvestmentAdvisoryFees` | 1 | 2% ░░░░░░░░░░ |
| `LiabilityForFuturePolicyBenefitInterestExpense` | 1 | 2% ░░░░░░░░░░ |
| `InterestExpenseShortTermBorrowings` | 1 | 2% ░░░░░░░░░░ |
| `InterestExpenseOtherShortTermBorrowings` | 1 | 2% ░░░░░░░░░░ |
| `InterestExpenseTradingLiabilities` | 1 | 2% ░░░░░░░░░░ |
| `InterestExpenseShortTermBorrowingsExcludingFederalFundsAndSecuritiesSoldUnderAgreementsToRepurchase` | 1 | 2% ░░░░░░░░░░ |
| `InterestExpenseJuniorSubordinatedDebentures` | 1 | 2% ░░░░░░░░░░ |

**⚠ No match in 2/50 companies:** *NEWS CORPORATION*, *GE Vernova Inc.*

---

## Depreciation

| Label | n/50 | Coverage |
|-------|------|---------|
| `AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment` | 42 | 84% ████████░░ |
| `DepreciationDepletionAndAmortization` | 37 | 74% ███████░░░ |
| `Depreciation` | 33 | 66% ███████░░░ |
| `DepreciationAndAmortization` | 24 | 48% █████░░░░░ |
| `PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization` | 14 | 28% ███░░░░░░░ |
| `PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAccumulatedDepreciationAndAmortization` | 12 | 24% ██░░░░░░░░ |
| `PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization` | 12 | 24% ██░░░░░░░░ |
| `DepreciationAmortizationAndAccretionNet` | 6 | 12% █░░░░░░░░░ |
| `OtherDepreciationAndAmortization` | 4 | 8% █░░░░░░░░░ |
| `RealEstateAndAccumulatedDepreciationInitialCostOfLand` | 3 | 6% █░░░░░░░░░ |
| `RealEstateAndAccumulatedDepreciationAmountOfEncumbrances` | 3 | 6% █░░░░░░░░░ |
| `RealEstateAndAccumulatedDepreciationCarryingAmountOfBuildingsAndImprovements` | 3 | 6% █░░░░░░░░░ |
| `RealEstateAccumulatedDepreciationRealEstateSold` | 3 | 6% █░░░░░░░░░ |
| `RealEstateAndAccumulatedDepreciationCarryingAmountOfLand` | 3 | 6% █░░░░░░░░░ |
| `RealEstateAccumulatedDepreciationDepreciationExpense` | 3 | 6% █░░░░░░░░░ |
| `RealEstateAccumulatedDepreciation` | 3 | 6% █░░░░░░░░░ |
| `RealEstateInvestmentPropertyAccumulatedDepreciation` | 3 | 6% █░░░░░░░░░ |
| `DepreciationNonproduction` | 3 | 6% █░░░░░░░░░ |
| `SECScheduleIIIRealEstateAccumulatedDepreciationDepreciationExpense` | 3 | 6% █░░░░░░░░░ |
| `RealEstateAndAccumulatedDepreciationInitialCostOfBuildingsAndImprovements` | 3 | 6% █░░░░░░░░░ |
| `SegmentReportingInformationDepreciationDepletionAndAmortizationExpense` | 3 | 6% █░░░░░░░░░ |
| `RealEstateAndAccumulatedDepreciationLifeUsedForDepreciation` | 2 | 4% ░░░░░░░░░░ |
| `DisposalGroupIncludingDiscontinuedOperationDepreciationAndAmortization` | 2 | 4% ░░░░░░░░░░ |
| `RealEstateAccumulatedDepreciationOtherDeductions` | 2 | 4% ░░░░░░░░░░ |
| `RestructuringReserveAcceleratedDepreciation` | 2 | 4% ░░░░░░░░░░ |
| `PropertySubjectToOrAvailableForOperatingLeaseAccumulatedDepreciation` | 2 | 4% ░░░░░░░░░░ |
| `AccumulatedDepreciationDepletionAndAmortizationExpensePropertyPlantAndEquipmentCurrentCharge` | 2 | 4% ░░░░░░░░░░ |
| `RealEstateAndAccumulatedDepreciationCarryingAmountOfLandAndBuildingsAndImprovements` | 2 | 4% ░░░░░░░░░░ |
| `RealEstateAndAccumulatedDepreciationCostsCapitalizedSubsequentToAcquisitionCarryingCosts` | 2 | 4% ░░░░░░░░░░ |
| `TaxBasisOfInvestmentsUnrealizedAppreciationDepreciationNet` | 2 | 4% ░░░░░░░░░░ |
| `CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization` | 1 | 2% ░░░░░░░░░░ |
| `CostOfGoodsSoldDepreciation` | 1 | 2% ░░░░░░░░░░ |
| `CostOfGoodsAndServicesSoldDepreciation` | 1 | 2% ░░░░░░░░░░ |
| `DepreciationAndAmortizationDiscontinuedOperations` | 1 | 2% ░░░░░░░░░░ |
| `CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization` | 1 | 2% ░░░░░░░░░░ |
| `SegmentReportingInformationDepreciationExpense` | 1 | 2% ░░░░░░░░░░ |
| `CostOfServicesDepreciationAndAmortization` | 1 | 2% ░░░░░░░░░░ |
| `CostOfGoodsSoldDepreciationAndAmortization` | 1 | 2% ░░░░░░░░░░ |
| `RealEstateAndAccumulatedDepreciationCostsCapitalizedSubsequentToAcquisitionImprovements` | 1 | 2% ░░░░░░░░░░ |
| `RealEstateAccumulatedDepreciationOtherAdditions` | 1 | 2% ░░░░░░░░░░ |
| `OperatingLeasesIncomeStatementDepreciationExpenseOnPropertySubjectToOrHeldForLease` | 1 | 2% ░░░░░░░░░░ |
| `IncomeTaxReconciliationNondeductibleExpenseDepreciation` | 1 | 2% ░░░░░░░░░░ |
| `CostDepreciationAmortizationAndDepletion` | 1 | 2% ░░░░░░░░░░ |
| `EffectiveIncomeTaxRateReconciliationNondeductibleExpenseDepreciation` | 1 | 2% ░░░░░░░░░░ |
| `ResultsOfOperationsDepreciationDepletionAndAmortizationAndValuationProvisions` | 1 | 2% ░░░░░░░░░░ |
| `CapitalizedCostsAccumulatedDepreciationDepletionAmortizationAndValuationAllowanceForRelatingToOilAndGasProducingActivities` | 1 | 2% ░░░░░░░░░░ |
| `IncomeTaxReconciliationNondeductibleExpenseDepreciationAndAmortization` | 1 | 2% ░░░░░░░░░░ |
| `RealEstateAndAccumulatedDepreciationAccumulatedDepreciation` | 1 | 2% ░░░░░░░░░░ |
| `PublicUtilitiesPropertyPlantAndEquipmentDisclosureOfCompositeDepreciationRateForPlantsInService` | 1 | 2% ░░░░░░░░░░ |
| `PublicUtilitiesPropertyPlantAndEquipmentAccumulatedDepreciation` | 1 | 2% ░░░░░░░░░░ |
| `InvestmentOwnedUnrecognizedUnrealizedAppreciationDepreciationNet` | 1 | 2% ░░░░░░░░░░ |
| `InvestmentOwnedUnrecognizedUnrealizedDepreciation` | 1 | 2% ░░░░░░░░░░ |
| `InvestmentOwnedUnrealizedAppreciationDepreciationNet` | 1 | 2% ░░░░░░░░░░ |

**✓ All 50 companies have at least one matching label.**

---

## Tax Expense

| Label | n/50 | Coverage |
|-------|------|---------|
| `IncomeTaxExpenseBenefit` | 49 | 98% ██████████ |
| `DeferredFederalIncomeTaxExpenseBenefit` | 45 | 90% █████████░ |
| `DeferredIncomeTaxExpenseBenefit` | 44 | 88% █████████░ |
| `DeferredStateAndLocalIncomeTaxExpenseBenefit` | 42 | 84% ████████░░ |
| `IncomeTaxReconciliationIncomeTaxExpenseBenefitAtFederalStatutoryIncomeTaxRate` | 41 | 82% ████████░░ |
| `DeferredForeignIncomeTaxExpenseBenefit` | 40 | 80% ████████░░ |
| `CurrentIncomeTaxExpenseBenefit` | 36 | 72% ███████░░░ |
| `FederalIncomeTaxExpenseBenefitContinuingOperations` | 13 | 26% ███░░░░░░░ |
| `StateAndLocalIncomeTaxExpenseBenefitContinuingOperations` | 13 | 26% ███░░░░░░░ |
| `ForeignIncomeTaxExpenseBenefitContinuingOperations` | 11 | 22% ██░░░░░░░░ |
| `IncomeTaxExpenseBenefitContinuingOperationsAdjustmentOfDeferredTaxAssetLiability` | 10 | 20% ██░░░░░░░░ |
| `TaxCutsAndJobsActOf2017MeasurementPeriodAdjustmentIncomeTaxExpenseBenefit` | 8 | 16% ██░░░░░░░░ |
| `TaxCutsAndJobsActOf2017IncomeTaxExpenseBenefit` | 8 | 16% ██░░░░░░░░ |
| `IncomeTaxExpenseBenefitContinuingOperations` | 8 | 16% ██░░░░░░░░ |
| `TaxCutsAndJobsActOf2017IncompleteAccountingProvisionalIncomeTaxExpenseBenefit` | 6 | 12% █░░░░░░░░░ |
| `TaxCutsAndJobsActOf2017ChangeInTaxRateIncomeTaxExpenseBenefit` | 5 | 10% █░░░░░░░░░ |
| `IncomeTaxExpenseBenefitIntraperiodTaxAllocation` | 4 | 8% █░░░░░░░░░ |
| `TaxCutsAndJobsActOf2017IncompleteAccountingChangeInTaxRateProvisionalIncomeTaxExpenseBenefit` | 3 | 6% █░░░░░░░░░ |
| `OtherIncomeTaxExpenseBenefitContinuingOperations` | 3 | 6% █░░░░░░░░░ |
| `IncomeTaxExpenseBenefitContinuingOperationsDiscontinuedOperationsExtraordinaryItems` | 2 | 4% ░░░░░░░░░░ |

**⚠ No match in 1/50 companies:** *BXP, INC.*

---

