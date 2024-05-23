using Plots, Statistics, CSV, DataFrames, LinearAlgebra, CurveFit, StatsBase, HypothesisTests

##
Data = CSV.File("Life Expectancy Data.csv") |> DataFrame
dropmissing!(Data, ["Life expectancy ", "Adult Mortality"])
#filter!(x->(log(x."Adult Mortality") > -0.05x."Life expectancy " .+ 7.5), Data)
Y = Data."Life expectancy "
X = Data."Adult Mortality"

m1 = findall(x->x<70, Y)
for i in m1
    if X[i] < 10
        X[i] = X[i]*100
    elseif X[i] < 100
        X[i] = X[i]*10
    end
end

m2 = findall(x->70<=x, Y)
for i in m2
    if X[i] <= 3 
        X[i] = X[i]*100
    elseif X[i] < 40
        X[i] = X[i]*10
    end
end
##
p = histogram(Y, label=false, ylabel="ilość elementów", xlabel="przedziały wartości", size=(680, 480))
savefig(p, "histogrmaZ.png")
##
p = boxplot(Y, ylabel="warotści", label=false, size=(680, 480))
savefig(p, "boxplotZ.png")
##
xs = LinRange(40, 100, 10^3)
p = plot(xs, ecdf(Y)(xs), xlabel="wartości", ylabel="dystrybuanta empiryczna", size=(680, 480), label=false)
savefig(p, "dystrybuantaZ.png")
##
mean(Y)
##
geomean(Y)
##
var(Y)
##
std(Y)
##
skewness(Y)
##
kurtosis(Y)

##
quantile(Y, 1/4)
##
median(Y)
##
quantile(Y, 3/4)
##
iqr(Y)
##
minimum(X)
##
maximum(X)
##
xs = LinRange(49, 730, 10^3)
p = scatter(X, Y, ms=1.6, size=(680, 480), label="dane", xlabel="śmiertelność wśród dorosłych", ylabel="średnia długość życia")
p = plot!(xs, (b0 .+ b1*xs).^2, lw=2.5, label="krzywa regresji: z = ($(round(b0, digits=4)) - $(round(-b1, digits=4))x)²")
savefig(p, "regresjaL.png")
##
xs = LinRange(60, 800, 10^3)
b, a = linear_fit(X, (Y).^(1/50))
scatter(X, Y.^(1/50))
plot!(xs, a*xs .+ b, lw=2)

##
scatter(X, Y, ms=1.6, size=(680, 480), label=false, xlabel="śmiertelność wśród dorosłych", ylabel="średnia długość życia")
xs = LinRange(60, 800, 10^3)
b, a = linear_fit(X, (Y).^(1/2))
##
Ym = sqrt.(Y)
b1 = sum(X.*(Ym .- mean(Ym)))/sum((X.-mean(X)).^2)
##
b0 = mean(Ym)-b1*mean(X)

## estymacja przedziałowa
n = length(Ym)
S = sqrt(1/(n-2)*sum((Ym.-(b0.+b1*X)).^2))
##
α = 0.05
A1 = b1-quantile(TDist(n-2), 1-α/2)*S/sqrt(sum((X.-mean(X)).^2))

##
B1 = b1+quantile(TDist(n-2), 1-α/2)*S/sqrt(sum((X.-mean(X)).^2))

##
A0 = b0-quantile(TDist(n-2), 1-α/2)*S*sqrt(1/n+mean(X)^2/sum((X.-mean(X)).^2))

##
B0 = b0+quantile(TDist(n-2), 1-α/2)*S*sqrt(1/n+mean(X)^2/sum((X.-mean(X)).^2))

##
xs = LinRange(49, 730, 10^3)
p = scatter(X, Ym, ms=1.6, size=(680, 480), label="dane", xlabel="śmiertelność wśród dorosłych", ylabel="pierwiastek średniej długości życia", c=1)
p = plot!(xs, B0 .+ B1*xs, lw=1, fillrange=A0 .+ A1*xs, c=2, alpha=0.5, label="obszar z możliwymi prostymi regresji")
p = plot!(xs, B0 .+ B1*xs, lw=1, c=2, label=false)
p = plot!(xs, A0 .+ A1*xs, lw=1, c=2, label=false)
savefig(p, "przedzialP.png")
##
xs = LinRange(49, 730, 10^3)
p = scatter(X, Y, ms=1.6, size=(680, 480), label="dane", xlabel="śmiertelność wśród dorosłych", ylabel="średnia długość życia", c=1)
p = plot!(xs, (B0 .+ B1*xs).^2, lw=1, fillrange=(A0 .+ A1*xs).^2, c=2, alpha=0.5, label="obszar z możliwymi krzywymi regresji")
p = plot!(xs, (B0 .+ B1*xs).^2, lw=1, c=2, label=false)
p = plot!(xs, (A0 .+ A1*xs).^2, lw=1, c=2, label=false)
savefig(p, "przedzialL.png")


## Ocena poziomu zależności
cor(X, Ym)
##
estYm = b0 .+ b1*X 
##
SST = sum((Ym .- mean(Ym)).^2)
##
SSE = sum((Ym .- estYm).^2)
##
SSR = sum((estYm .- mean(Ym)).^2)
##

## predykcja
F = findall((x)->x>=500, X)
UF = findall((x)->x<500, X)
##
pX = X[UF]
pY = Ym[UF]
rX = X[F]
rY = Ym[F]
##
xs = LinRange(49, 730, 10^3)
p = scatter(pX, pY, ms=1.7, markerstrokewidth=0.2, size=(680, 480), label="dane treningowe", xlabel="śmiertelność wśród dorosłych", ylabel="pierwiastek średniej długości życia")
p = scatter!(rX, rY, markerstrokewidth=0.2, ms=2.4, label="dane testowe", c=:lime)
savefig(p, "podzial.png")
##
pb1 = sum(pX.*(pY .- mean(pY)))/sum((pX.-mean(pX)).^2)

##
pb0 = mean(pY)-b1*mean(pX)

##
xs = LinRange(49, 730, 10^3)
p = scatter(pX, pY, ms=1.7, markerstrokewidth=0.2, size=(680, 480), label="dane treningowe", xlabel="śmiertelność wśród dorosłych", ylabel="pierwiastek średniej długości życia")
p = scatter!(rX, rY, markerstrokewidth=0.2, ms=2.4, label="dane testowe", c="lime")
p = plot!(xs, pb0 .+ pb1*xs, lw=2.5, c=2, label="krzywa regresji: y = $(round(pb0, digits=4)) - $(round(-pb1, digits=4))x")
savefig(p, "regpodzial.png")
##
α = 0.05
k = length(pX)
pS = sqrt(1/(k-2)*sum((pY.-(pb0.+pb1*pX)).^2))

pA = Vector{Float64}(undef, length(rX))
pB = Vector{Float64}(undef, length(rX))
for (i, v) in enumerate(rX)
    q = quantile(TDist(k-2), 1-α/2)*pS*sqrt(1+1/k+(v - mean(pX))^2/sum((pX.-mean(pX).^2)))
    pA[i] = pb0+pb1*v-q
    pB[i] = pb0+pb1*v+q
end
##
xs = LinRange(49, 730, 10^3)
p = scatter(pX, pY, ms=1.7, markerstrokewidth=0.2, size=(680, 480), label="dane treningowe", xlabel="śmiertelność wśród dorosłych", ylabel="pierwiastek średniej długości życia")
p = scatter!(rX, rY, markerstrokewidth=0.2, ms=2.4, label="dane testowe", c="lime")
p = plot!(rX, pA, c=2, label=false)
p = plot!(rX, pB, c=2, label=false)
p = plot!(sort(rX), sort(pA, rev=true), fillrange=sort(pB, rev=true), c=2, alpha=0.5, label="przedział ufności dla predykowanych obserwacji")
savefig(p, "predykcjaL.png")
##
xs = LinRange(49, 730, 10^3)
p = scatter(pX, pY.^2, ms=1.7, markerstrokewidth=0.2, size=(680, 480), label="dane treningowe", xlabel="śmiertelność wśród dorosłych", ylabel="średnia długość życia")
p = scatter!(rX, rY.^2, markerstrokewidth=0.2, ms=2.4, label="dane testowe", c="lime")
p = plot!(rX, pA.^2, c=2, label=false)
p = plot!(rX, pB.^2, c=2, label=false)
p = plot!(sort(rX), sort(pA.^2, rev=true), fillrange=sort(pB.^2, rev=true), c=2, alpha=0.5, label="przedział ufności dla predykowanych obserwacji")
savefig(p, "predykcjaP.png")

##
n = length(Y)
Ym = sqrt.(Y)
## analiza residuów
e = Ym.-(b0.+b1*X)
##
p1 = scatter(X, e, size=(680, 480), c=:crimson, ms=2.5, markerstrokewidth=0.5, label=false, ylabel="residua", xlabel="śmiertelność wśród dorosłych (zmienna niezależna)")
#savefig(p1, "residua1.png")
##
p2 = scatter(e, c=:crimson, size=(680, 480), ms=2.5, markerstrokewidth=0.5, label=false, ylabel="residua", xlabel="numer próby")
#savefig(p2, "residua2.png")

##
n
h = div(n,2)
OneSampleTTest(e[h+1:end])
##
mean(e[h+1:end])
##
mean(e[1:h])
##
var(e[1:h])
##
var(e[h+1:end])
##
VarianceFTest(e[1:h], e[h+1:end])
##
#A = autocor(e, 0:40)
p = scatter(0:40, A, line=:stem, label=false, xlabel="h", ylabel="ρ̂(h)")
p = scatter!(0:40, A, c=1, ms=3.5, label=false, size=(680, 480))
#savefig(p, "ACFV")

##
xs=LinRange(-0.04, 0.04, 10^3)
histogram(A, normed=true)
plot!(xs, pdf.(Normal(mean(A), std(A)), xs))

##
μ = mean(e)
σ = std(e)
xs = LinRange(-1, 1, 10^3)
p = histogram(e, normed=true, xlabel="x", ylabel="f(x)", ylim=[0, 3.5], size=(680, 480), label="histogram przybliżający gęstość residuów")
p = plot!(xs, pdf.(Normal(μ, σ), xs), lw=2, label="gęstość f(x) rozkładu N($(round(μ, digits=3)), $(round(σ^2, digits=3)))")
savefig(p, "resHist")
##
p = plot(xs, ecdf(e)(xs), lw=1.6, label="dystrybuanta empiryczna residuów", xlabel="x", ylabel="F(x)", size=(680, 480))
p = plot!(xs, cdf.(Normal(μ, σ), xs), lw=1.6, label="dystrybuanta F(x) rozkładu N($(round(μ, digits=3)), $(round(σ^2, digits=3)))")
savefig(p, "resDystr.png")
##
Pingouin.normality(e)