from modeling import *

def get_data(red, r, ro):
  if red == 'Coding' and r == 2:
    d_l= [
      0,
      10.0,
      12.593969317272713,
      15.860806316440653,
      19.975050809645882,
      25.156517700764358,
      31.682041205285422,
      39.900265484793444,
      50.250271926652381,
      63.285038282887051,
      79.700983037710856,
      100.37517349334034,
      126.41218551910536,
      159.2031185756999,
      200.4999190556494,
      252.50898287025086,
      318.00903826036807,
      400.49960704664795,
      504.38797627252632,
      635.22466971774736,
      800.00000000000034]
    if ro == 0.5:
      sim_ET_l= [
        23.944456881887493,
        23.924892602441275,
        23.198682864860814,
        22.691584392480358,
        22.43986715618567,
        21.484092438055228,
        20.865332478622609,
        19.942906210502354,
        18.742643017954588,
        17.581513774102316,
        17.006839221519282,
        16.76659262987102,
        16.639109966817124,
        16.586265906547204,
        16.545301742454676,
        16.527165380235559,
        16.532973144177312,
        16.516654681711554,
        16.512261306150446,
        16.513146732067277,
        16.519155806661054]
      sim_StdT_l= [
        0.06927220206984129,
        0.066383984163648727,
        0.054677440467456026,
        0.051195104426943347,
        0.060353655766260651,
        0.050198304331823228,
        0.048874884443081454,
        0.061028724349391827,
        0.041710371443432655,
        0.041133775827590306,
        0.040837268114482309,
        0.030315885038746861,
        0.044545820068631441,
        0.033391723365472109,
        0.028332550550612932,
        0.034004456750220229,
        0.04248290397635264,
        0.029084688703680254,
        0.021380115583299084,
        0.026215484290762297,
        0.042044609281664251]
    elif ro == 0.6:
      sim_ET_l= [
        23.931077243509911,
        23.945970074296561,
        23.195529686663441,
        22.713596572569806,
        22.417791645548856,
        21.463525508648313,
        20.884608690248747,
        19.989085243463336,
        18.956394465310314,
        18.611599783577979,
        19.172396592226285,
        20.478952853905994,
        21.744695151782302,
        22.348133148867884,
        23.2498557865332,
        23.221473695649589,
        23.180007050179913,
        24.328278599940276,
        24.106399344358429,
        24.272833073439863,
        23.290656458841653]
      sim_StdT_l= [
        0.046742590489707365,
        0.047290679328663716,
        0.05825449358458816,
        0.063626048821668377,
        0.068764886891555341,
        0.055094076294105031,
        0.06188012976452037,
        0.040082905578311068,
        0.043025069492885003,
        0.10670941000401758,
        0.23863167379601213,
        0.53319554712106643,
        0.88313833573555423,
        0.98532769106087725,
        1.0272176643163009,
        1.1032863039287444,
        1.4435361139075793,
        1.8752070218105232,
        2.0805024264753644,
        2.0009593912669446,
        1.7699161273767812]
    return d_l, sim_ET_l, sim_StdT_l

def plot_ET_wrt_d():
  red, r = 'Coding', 2
  ro = 0.5
  log(INFO, "red= {}, r= {}, ro= {}".format(red, r, ro) )
  d_l, sim_ET_l, sim_StdT_l = get_data(red, r, ro)
  blog(len_d_l=len(d_l), len_sim_ET_l=len(sim_ET_l), len_sim_StdT_l=len(sim_StdT_l) )
  
  ET_wMGc_l, approx_ET_wMGc_l = [], []
  for d in d_l:
    ET_wMGc, EW_wMGc, Prqing_wMGc = ET_EW_Prqing_pareto_wMGc(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
    approx_ET_wMGc, approx_EW_wMGc, approx_Prqing_wMGc = approx_ET_EW_Prqing_pareto_wMGc(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
    ET_wMGc_l.append(ET_wMGc)
    approx_ET_wMGc_l.append(approx_ET_wMGc)
  
  plot.errorbar(d_l, sim_ET_l, yerr=sim_StdT_l, label='Sim', c=next(darkcolor_c), marker=next(marker_c), ls=':')
  plot.plot(d_l, ET_wMGc_l, label='M/G/c model', c=next(darkcolor_c), marker=next(marker_c), ls=':')
  plot.plot(d_l, approx_ET_wMGc_l, label='Asymptotic M/G/c model', c=next(darkcolor_c), marker=next(marker_c), ls=':')
  fontsize = 14
  prettify(plot.gca() )
  plot.legend(loc='best', framealpha=0.5, fontsize=12)
  plot.xscale('log')
  plot.xlabel('d', fontsize=fontsize)
  plot.ylabel('E[T]', fontsize=fontsize)
  
  plot.title(r'$N= {}$, $Cap= {}$, $\rho_0= {}$, $r= {}$'.format(N, Cap, ro, r) + '\n' \
    + r'$k \sim${}, $L \sim${}, $Sl \sim${}'.format(k.to_latex(), L.to_latex(), Sl.to_latex() ) )
  plot.gca().title.set_position([.5, 1.05] )
  fig = plot.gcf()
  fig.set_size_inches(5, 5)
  plot.savefig('plot_ET_wrt_d_r{}_ro{}.png'.format(r, ro), bbox_inches='tight')
  fig.clear()
  log(INFO, "done.")

if __name__ == "__main__":
  N, Cap = 20, 10
  k = BZipf(1, 5) # DUniform(1, 1)
  R = Uniform(1, 1)
  b, beta = 10, 4
  L = Pareto(b, beta) # TPareto(10, 10**6, 4)
  a, alpha = 1, 3 # 1, 4
  Sl = Pareto(a, alpha) # Uniform(1, 1)
  def alpha_gen(ro):
    return alpha
  
  # ar = round(ar_for_ro(ro, N, Cap, k, R, L, S), 2)
  # sinfo_m = {
  #   'njob': 5000*N,
  #   'nworker': N, 'wcap': Cap, 'ar': ar,
  #   'k_rv': k,
  #   'reqed_rv': R,
  #   'lifetime_rv': L,
  #   'straggle_m': {'slowdown': lambda load: S.sample() } }
  # mapping_m = {'type': 'spreading'}
  # sching_m = {'type': 'expand_if_totaldemand_leq', 'r': r, 'threshold': None}
  
  # u = 40*L.mean()*Sl.mean()
  # for d in [0, *np.logspace(math.log10(l), math.log10(u), 20) ]:
  
  plot_ET_wrt_d()
